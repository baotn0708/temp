from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .gru import (
    GRUDataset,
    RatioSplitConfig,
    SplitArrays,
    TARGET_DIM,
    TARGET_KEYS,
    _metrics_from_price_arrays,
    _scaled_predictions,
    gather_split_arrays,
    load_asset_dataset,
    metrics_bundle_to_rows,
    reconstruct_prices_from_logreturns,
    resolve_datasets,
)
from .sequence_common import FeatureScaler, TrainConfig, save_json, set_seed


@dataclass
class LSTMConfig:
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 1e-3


class LSTMMultiOutputRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.10,
        output_dim: int = TARGET_DIM,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.direction_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, X_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(X_seq)
        state = out[:, -1]
        pred = self.head(state)
        direction_logit = self.direction_head(state)
        return pred, direction_logit


def default_lstm_candidates() -> List[LSTMConfig]:
    return [
        LSTMConfig(hidden_dim=24, num_layers=1, dropout=0.10, lr=1e-3),
        LSTMConfig(hidden_dim=32, num_layers=1, dropout=0.10, lr=1e-3),
        LSTMConfig(hidden_dim=48, num_layers=1, dropout=0.00, lr=1e-3),
        LSTMConfig(hidden_dim=32, num_layers=2, dropout=0.10, lr=5e-4),
    ]


def _make_model(input_dim: int, cfg: LSTMConfig) -> LSTMMultiOutputRegressor:
    return LSTMMultiOutputRegressor(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        output_dim=TARGET_DIM,
    )


def predict_lstm(
    model: LSTMMultiOutputRegressor,
    dataset: GRUDataset,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    preds, ys, scales = [], [], []
    with torch.no_grad():
        for X_seq, y_z, scale in loader:
            pred_z, _ = model(X_seq)
            preds.append(pred_z.cpu().numpy())
            ys.append(y_z.cpu().numpy())
            scales.append(scale.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0), np.concatenate(scales, axis=0)


def evaluate_lstm(
    model: LSTMMultiOutputRegressor,
    dataset: GRUDataset,
    current_ohlc: np.ndarray,
    actual_next_ohlc: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, object]:
    pred_z, y_z, scale = predict_lstm(model, dataset, batch_size=batch_size)
    _y_true, y_pred = _scaled_predictions(y_z, pred_z, scale)
    pred_next_ohlc = reconstruct_prices_from_logreturns(current_ohlc, y_pred)
    return _metrics_from_price_arrays(current_ohlc, actual_next_ohlc, pred_next_ohlc)


def train_lstm_with_early_stopping(
    model: LSTMMultiOutputRegressor,
    train_ds: GRUDataset,
    val_ds: GRUDataset,
    val_current_ohlc: np.ndarray,
    val_next_ohlc: np.ndarray,
    lstm_cfg: LSTMConfig,
    train_cfg: TrainConfig,
) -> Tuple[LSTMMultiOutputRegressor, Dict[str, object], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lstm_cfg.lr, weight_decay=train_cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    best_score = float("inf")
    best_state = None
    best_epoch = 1
    bad_epochs = 0

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        for X_seq, y_z, _scale in train_loader:
            pred_z, direction_logit = model(X_seq)
            loss = F.smooth_l1_loss(pred_z, y_z)
            loss = loss + train_cfg.direction_loss_weight * F.binary_cross_entropy_with_logits(
                direction_logit, (y_z > 0).float()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = evaluate_lstm(
            model,
            val_ds,
            current_ohlc=val_current_ohlc,
            actual_next_ohlc=val_next_ohlc,
            batch_size=train_cfg.batch_size,
        )
        if float(metrics["selection_score"]) < best_score - 1e-5:
            best_score = float(metrics["selection_score"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate_lstm(
        model,
        val_ds,
        current_ohlc=val_current_ohlc,
        actual_next_ohlc=val_next_ohlc,
        batch_size=train_cfg.batch_size,
    )
    return model, final_metrics, best_epoch


def fit_lstm_fixed_epochs(
    model: LSTMMultiOutputRegressor,
    train_ds: GRUDataset,
    lstm_cfg: LSTMConfig,
    train_cfg: TrainConfig,
    epochs: int,
) -> LSTMMultiOutputRegressor:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lstm_cfg.lr, weight_decay=train_cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    for _ in range(max(1, epochs)):
        model.train()
        for X_seq, y_z, _scale in train_loader:
            pred_z, direction_logit = model(X_seq)
            loss = F.smooth_l1_loss(pred_z, y_z)
            loss = loss + train_cfg.direction_loss_weight * F.binary_cross_entropy_with_logits(
                direction_logit, (y_z > 0).float()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def tune_lstm_config(
    split: SplitArrays,
    train_cfg: TrainConfig,
    candidates: Sequence[LSTMConfig],
    seed: int,
) -> Tuple[LSTMConfig, int, Dict[str, object]]:
    if not split.has_val:
        raise ValueError("Validation split is required to tune LSTM configs.")

    scaler = FeatureScaler().fit(split.X_train_raw)
    X_train = scaler.transform(split.X_train_raw)
    X_val = scaler.transform(split.X_val_raw)

    train_ds = GRUDataset(X_train, split.y_train_z, split.scale_train)
    val_ds = GRUDataset(X_val, split.y_val_z, split.scale_val)

    best_cfg = candidates[0]
    best_epoch = 1
    best_metrics = {"by_target": {}, "selection_score": float("inf")}

    for cfg in candidates:
        set_seed(seed)
        model = _make_model(input_dim=X_train.shape[-1], cfg=cfg)
        _, metrics, best_epoch_here = train_lstm_with_early_stopping(
            model,
            train_ds,
            val_ds,
            val_current_ohlc=split.current_ohlc_val,
            val_next_ohlc=split.next_ohlc_val,
            lstm_cfg=cfg,
            train_cfg=train_cfg,
        )
        if float(metrics["selection_score"]) < float(best_metrics["selection_score"]) - 1e-5:
            best_cfg = cfg
            best_epoch = best_epoch_here
            best_metrics = metrics

    return best_cfg, best_epoch, best_metrics


def fit_and_eval_lstm(
    split: SplitArrays,
    lstm_cfg: LSTMConfig,
    train_cfg: TrainConfig,
    seed: int,
    epochs: int | None = None,
) -> Dict[str, object]:
    train_blocks = [split.X_train_raw]
    y_blocks = [split.y_train_z]
    scale_blocks = [split.scale_train]

    if split.has_val:
        train_blocks.append(split.X_val_raw)
        y_blocks.append(split.y_val_z)
        scale_blocks.append(split.scale_val)

    X_train_raw = np.concatenate(train_blocks, axis=0)
    y_train_z = np.concatenate(y_blocks, axis=0)
    scale_train = np.concatenate(scale_blocks, axis=0)

    scaler = FeatureScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(split.X_test_raw)

    train_ds = GRUDataset(X_train, y_train_z, scale_train)
    test_ds = GRUDataset(X_test, split.y_test_z, split.scale_test)

    set_seed(seed)
    model = _make_model(input_dim=X_train.shape[-1], cfg=lstm_cfg)
    model = fit_lstm_fixed_epochs(model, train_ds, lstm_cfg, train_cfg, epochs=epochs or train_cfg.max_epochs)
    return evaluate_lstm(
        model,
        test_ds,
        current_ohlc=split.current_ohlc_test,
        actual_next_ohlc=split.next_ohlc_test,
        batch_size=train_cfg.batch_size,
    )


def fit_final_lstm(
    split: SplitArrays,
    lstm_cfg: LSTMConfig,
    train_cfg: TrainConfig,
    seed: int,
    epochs: int | None = None,
) -> Tuple[LSTMMultiOutputRegressor, FeatureScaler, Dict[str, object]]:
    train_blocks = [split.X_train_raw]
    y_blocks = [split.y_train_z]
    scale_blocks = [split.scale_train]

    if split.has_val:
        train_blocks.append(split.X_val_raw)
        y_blocks.append(split.y_val_z)
        scale_blocks.append(split.scale_val)

    X_train_raw = np.concatenate(train_blocks, axis=0)
    y_train_z = np.concatenate(y_blocks, axis=0)
    scale_train = np.concatenate(scale_blocks, axis=0)

    scaler = FeatureScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(split.X_test_raw)

    train_ds = GRUDataset(X_train, y_train_z, scale_train)
    test_ds = GRUDataset(X_test, split.y_test_z, split.scale_test)

    set_seed(seed)
    model = _make_model(input_dim=X_train.shape[-1], cfg=lstm_cfg)
    model = fit_lstm_fixed_epochs(model, train_ds, lstm_cfg, train_cfg, epochs=epochs or train_cfg.max_epochs)
    test_metrics = evaluate_lstm(
        model,
        test_ds,
        current_ohlc=split.current_ohlc_test,
        actual_next_ohlc=split.next_ohlc_test,
        batch_size=train_cfg.batch_size,
    )
    return model, scaler, test_metrics


def select_lstm_config(
    split: SplitArrays,
    train_cfg: TrainConfig,
    tuning_seed: int,
    fast: bool = False,
    fixed_lstm_cfg: LSTMConfig | None = None,
) -> Tuple[LSTMConfig, int, Dict[str, object] | None]:
    if split.has_val:
        if fixed_lstm_cfg is not None:
            return fixed_lstm_cfg, train_cfg.max_epochs, None

        candidates = default_lstm_candidates()
        if fast:
            candidates = candidates[:2]
        selected_cfg, selected_epoch, val_metrics = tune_lstm_config(
            split=split,
            train_cfg=train_cfg,
            candidates=candidates,
            seed=tuning_seed,
        )
        return selected_cfg, selected_epoch, val_metrics

    if fixed_lstm_cfg is None:
        raise ValueError(
            "Split 7/3 has no validation set. To keep the protocol leak-free, "
            "you must pass an explicit fixed LSTM config."
        )
    return fixed_lstm_cfg, train_cfg.max_epochs, None


def benchmark_single_dataset(
    dataset_name: str,
    path: str | Path,
    split_cfg: RatioSplitConfig,
    train_cfg: TrainConfig,
    eval_seeds: Iterable[int],
    tuning_seed: int,
    fast: bool = False,
    fixed_lstm_cfg: LSTMConfig | None = None,
) -> Dict[str, object]:
    asset = load_asset_dataset(path=path, name=dataset_name, split_cfg=split_cfg)
    split = gather_split_arrays(asset, split_cfg)

    selected_cfg, selected_epoch, val_metrics = select_lstm_config(
        split=split,
        train_cfg=train_cfg,
        tuning_seed=tuning_seed,
        fast=fast,
        fixed_lstm_cfg=fixed_lstm_cfg,
    )

    rows: List[Dict[str, float | int | str]] = []
    metrics_by_seed: Dict[str, Dict[str, object]] = {}
    for seed in eval_seeds:
        metrics = fit_and_eval_lstm(
            split=split,
            lstm_cfg=selected_cfg,
            train_cfg=train_cfg,
            seed=int(seed),
            epochs=selected_epoch,
        )
        metrics_by_seed[str(int(seed))] = metrics
        rows.extend(
            metrics_bundle_to_rows(
                dataset_name=dataset_name,
                seed=int(seed),
                split_ratio=split_cfg.split_ratio,
                metrics_bundle=metrics,
            )
        )

    for row in rows:
        row["model"] = "lstm_only"

    return {
        "dataset": dataset_name,
        "path": str(path),
        "rows": rows,
        "metrics_by_seed": metrics_by_seed,
        "selected_config": asdict(selected_cfg),
        "selected_epoch": int(selected_epoch),
        "val_metrics": val_metrics,
        "sample_counts": split.sample_counts(),
        "has_val": split.has_val,
    }


def save_training_artifacts(
    output_dir: str | Path,
    dataset_name: str,
    split_cfg: RatioSplitConfig,
    train_cfg: TrainConfig,
    lstm_cfg: LSTMConfig,
    selected_epoch: int,
    sample_counts: Dict[str, int],
    val_metrics: Dict[str, object] | None,
    test_metrics: Dict[str, object],
    scaler: FeatureScaler,
    model: LSTMMultiOutputRegressor,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "lstm_state_dict.pt")
    np.savez(output_path / "lstm_scaler.npz", mean=scaler.scaler.mean_, scale=scaler.scaler.scale_)

    payload = {
        "dataset": dataset_name,
        "model": "lstm_only",
        "target_definition": "model predicts next-day log-return for Open, High, Low, Close",
        "evaluation_definition": "benchmark/train metrics are computed on algebraically reconstructed next-day OHLC prices",
        "target_order": TARGET_KEYS,
        "split_config": asdict(split_cfg),
        "train_config": asdict(train_cfg),
        "selected_config": asdict(lstm_cfg),
        "selected_epoch": int(selected_epoch),
        "sample_counts": sample_counts,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(output_path / "lstm_training_summary.json", payload)
