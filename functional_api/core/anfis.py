from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .gru import (
    DEFAULT_DATASETS,
    RatioSplitConfig,
    SplitArrays,
    TARGET_DIM,
    TARGET_KEYS,
    _scaled_predictions,
    gather_split_arrays,
    load_asset_dataset,
    reconstruct_prices_from_logreturns,
    _metrics_from_price_arrays,
    resolve_datasets,
)
from .sequence_common import FeatureScaler, TrainConfig, save_json, set_seed, summarize_windows


@dataclass
class ANFISConfig:
    n_rules: int = 6
    lr: float = 1e-3


class ANFISDataset(Dataset):
    def __init__(self, X_summary: np.ndarray, y_z: np.ndarray, scale: np.ndarray) -> None:
        self.X = torch.tensor(X_summary, dtype=torch.float32)
        self.y = torch.tensor(y_z, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.scale[idx]


class ANFISMultiOutputRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_rules: int = 6,
        output_dim: int = TARGET_DIM,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.centers = nn.Parameter(torch.randn(n_rules, input_dim) * 0.15)
        self.log_scales = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.rule_bias = nn.Parameter(torch.zeros(n_rules))
        self.consequent_weight = nn.Parameter(torch.randn(n_rules, input_dim, output_dim) * 0.05)
        self.consequent_bias = nn.Parameter(torch.zeros(n_rules, output_dim))
        self.direction_head = nn.Linear(input_dim, output_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def fuzzy_logits(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        scales = F.softplus(self.log_scales).unsqueeze(0) + 1e-3
        return -0.5 * ((diff / scales) ** 2).sum(dim=-1) + self.rule_bias

    def forward(self, x_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_norm(x_summary)
        gate = F.softmax(self.fuzzy_logits(x) / (F.softplus(self.temperature) + 1e-3), dim=-1)
        rule_outputs = torch.einsum("bd,rdo->bro", x, self.consequent_weight) + self.consequent_bias.unsqueeze(0)
        pred = (gate.unsqueeze(-1) * rule_outputs).sum(dim=1)
        direction_logit = self.direction_head(x)
        return pred, direction_logit, gate


def default_anfis_candidates() -> List[ANFISConfig]:
    return [
        ANFISConfig(n_rules=4, lr=1e-3),
        ANFISConfig(n_rules=6, lr=1e-3),
        ANFISConfig(n_rules=8, lr=1e-3),
        ANFISConfig(n_rules=6, lr=5e-4),
    ]


def _make_model(input_dim: int, cfg: ANFISConfig) -> ANFISMultiOutputRegressor:
    return ANFISMultiOutputRegressor(
        input_dim=input_dim,
        n_rules=cfg.n_rules,
        output_dim=TARGET_DIM,
    )


def _prepare_summary_views(split: SplitArrays):
    scaler = FeatureScaler().fit(split.X_train_raw)
    X_train = scaler.transform(split.X_train_raw)
    X_val = scaler.transform(split.X_val_raw) if split.has_val else np.empty((0,), dtype=np.float32)
    X_test = scaler.transform(split.X_test_raw)

    S_train = summarize_windows(X_train)
    S_val = summarize_windows(X_val) if split.has_val else np.empty((0, S_train.shape[1]), dtype=np.float32)
    S_test = summarize_windows(X_test)
    return scaler, S_train, S_val, S_test


def predict_anfis(
    model: ANFISMultiOutputRegressor,
    dataset: ANFISDataset,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    preds, ys, scales = [], [], []
    with torch.no_grad():
        for X_summary, y_z, scale in loader:
            pred_z, _, _ = model(X_summary)
            preds.append(pred_z.cpu().numpy())
            ys.append(y_z.cpu().numpy())
            scales.append(scale.cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0), np.concatenate(scales, axis=0)


def evaluate_anfis(
    model: ANFISMultiOutputRegressor,
    dataset: ANFISDataset,
    current_ohlc: np.ndarray,
    actual_next_ohlc: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, object]:
    pred_z, y_z, scale = predict_anfis(model, dataset, batch_size=batch_size)
    _y_true, y_pred = _scaled_predictions(y_z, pred_z, scale)
    pred_next_ohlc = reconstruct_prices_from_logreturns(current_ohlc, y_pred)
    return _metrics_from_price_arrays(current_ohlc, actual_next_ohlc, pred_next_ohlc)


def train_anfis_with_early_stopping(
    model: ANFISMultiOutputRegressor,
    train_ds: ANFISDataset,
    val_ds: ANFISDataset,
    val_current_ohlc: np.ndarray,
    val_next_ohlc: np.ndarray,
    anfis_cfg: ANFISConfig,
    train_cfg: TrainConfig,
) -> Tuple[ANFISMultiOutputRegressor, Dict[str, object], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=anfis_cfg.lr, weight_decay=train_cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    best_score = float("inf")
    best_state = None
    best_epoch = 1
    bad_epochs = 0

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        for X_summary, y_z, _scale in train_loader:
            pred_z, direction_logit, gate = model(X_summary)
            loss = F.smooth_l1_loss(pred_z, y_z)
            loss = loss + train_cfg.direction_loss_weight * F.binary_cross_entropy_with_logits(
                direction_logit, (y_z > 0).float()
            )
            entropy = -(gate * gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
            loss = loss - train_cfg.entropy_bonus * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = evaluate_anfis(
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
    final_metrics = evaluate_anfis(
        model,
        val_ds,
        current_ohlc=val_current_ohlc,
        actual_next_ohlc=val_next_ohlc,
        batch_size=train_cfg.batch_size,
    )
    return model, final_metrics, best_epoch


def fit_anfis_fixed_epochs(
    model: ANFISMultiOutputRegressor,
    train_ds: ANFISDataset,
    anfis_cfg: ANFISConfig,
    train_cfg: TrainConfig,
    epochs: int,
) -> ANFISMultiOutputRegressor:
    optimizer = torch.optim.AdamW(model.parameters(), lr=anfis_cfg.lr, weight_decay=train_cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    for _ in range(max(1, epochs)):
        model.train()
        for X_summary, y_z, _scale in train_loader:
            pred_z, direction_logit, gate = model(X_summary)
            loss = F.smooth_l1_loss(pred_z, y_z)
            loss = loss + train_cfg.direction_loss_weight * F.binary_cross_entropy_with_logits(
                direction_logit, (y_z > 0).float()
            )
            entropy = -(gate * gate.clamp_min(1e-8).log()).sum(dim=-1).mean()
            loss = loss - train_cfg.entropy_bonus * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def tune_anfis_config(
    split: SplitArrays,
    train_cfg: TrainConfig,
    candidates: Sequence[ANFISConfig],
    seed: int,
) -> Tuple[ANFISConfig, int, Dict[str, object]]:
    if not split.has_val:
        raise ValueError("Validation split is required to tune ANFIS configs.")

    _scaler, S_train, S_val, _S_test = _prepare_summary_views(split)
    train_ds = ANFISDataset(S_train, split.y_train_z, split.scale_train)
    val_ds = ANFISDataset(S_val, split.y_val_z, split.scale_val)

    best_cfg = candidates[0]
    best_epoch = 1
    best_metrics = {"by_target": {}, "selection_score": float("inf")}

    for cfg in candidates:
        set_seed(seed)
        model = _make_model(input_dim=S_train.shape[-1], cfg=cfg)
        _, metrics, best_epoch_here = train_anfis_with_early_stopping(
            model,
            train_ds,
            val_ds,
            val_current_ohlc=split.current_ohlc_val,
            val_next_ohlc=split.next_ohlc_val,
            anfis_cfg=cfg,
            train_cfg=train_cfg,
        )
        if float(metrics["selection_score"]) < float(best_metrics["selection_score"]) - 1e-5:
            best_cfg = cfg
            best_epoch = best_epoch_here
            best_metrics = metrics

    return best_cfg, best_epoch, best_metrics


def fit_and_eval_anfis(
    split: SplitArrays,
    anfis_cfg: ANFISConfig,
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

    S_train = summarize_windows(X_train)
    S_test = summarize_windows(X_test)

    train_ds = ANFISDataset(S_train, y_train_z, scale_train)
    test_ds = ANFISDataset(S_test, split.y_test_z, split.scale_test)

    set_seed(seed)
    model = _make_model(input_dim=S_train.shape[-1], cfg=anfis_cfg)
    model = fit_anfis_fixed_epochs(model, train_ds, anfis_cfg, train_cfg, epochs=epochs or train_cfg.max_epochs)
    return evaluate_anfis(
        model,
        test_ds,
        current_ohlc=split.current_ohlc_test,
        actual_next_ohlc=split.next_ohlc_test,
        batch_size=train_cfg.batch_size,
    )


def fit_final_anfis(
    split: SplitArrays,
    anfis_cfg: ANFISConfig,
    train_cfg: TrainConfig,
    seed: int,
    epochs: int | None = None,
) -> Tuple[ANFISMultiOutputRegressor, FeatureScaler, Dict[str, object]]:
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

    S_train = summarize_windows(X_train)
    S_test = summarize_windows(X_test)

    train_ds = ANFISDataset(S_train, y_train_z, scale_train)
    test_ds = ANFISDataset(S_test, split.y_test_z, split.scale_test)

    set_seed(seed)
    model = _make_model(input_dim=S_train.shape[-1], cfg=anfis_cfg)
    model = fit_anfis_fixed_epochs(model, train_ds, anfis_cfg, train_cfg, epochs=epochs or train_cfg.max_epochs)
    test_metrics = evaluate_anfis(
        model,
        test_ds,
        current_ohlc=split.current_ohlc_test,
        actual_next_ohlc=split.next_ohlc_test,
        batch_size=train_cfg.batch_size,
    )
    return model, scaler, test_metrics


def select_anfis_config(
    split: SplitArrays,
    train_cfg: TrainConfig,
    tuning_seed: int,
    fast: bool = False,
    fixed_anfis_cfg: ANFISConfig | None = None,
) -> Tuple[ANFISConfig, int, Dict[str, object] | None]:
    if split.has_val:
        if fixed_anfis_cfg is not None:
            return fixed_anfis_cfg, train_cfg.max_epochs, None

        candidates = default_anfis_candidates()
        if fast:
            candidates = candidates[:2]
        selected_cfg, selected_epoch, val_metrics = tune_anfis_config(
            split=split,
            train_cfg=train_cfg,
            candidates=candidates,
            seed=tuning_seed,
        )
        return selected_cfg, selected_epoch, val_metrics

    if fixed_anfis_cfg is None:
        raise ValueError(
            "Split 7/3 has no validation set. To keep the protocol leak-free, "
            "you must pass an explicit fixed ANFIS config."
        )
    return fixed_anfis_cfg, train_cfg.max_epochs, None


def metrics_bundle_to_rows(
    dataset_name: str,
    seed: int,
    split_ratio: str,
    metrics_bundle: Dict[str, object],
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    by_target = metrics_bundle["by_target"]
    for target_key in TARGET_KEYS:
        metrics = by_target[target_key]
        rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "model": "anfis_only",
                "split_ratio": split_ratio,
                "target": target_key,
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
                "sign_acc": float(metrics["sign_acc"]),
            }
        )
    return rows


def benchmark_single_dataset(
    dataset_name: str,
    path: str | Path,
    split_cfg: RatioSplitConfig,
    train_cfg: TrainConfig,
    eval_seeds: Iterable[int],
    tuning_seed: int,
    fast: bool = False,
    fixed_anfis_cfg: ANFISConfig | None = None,
) -> Dict[str, object]:
    asset = load_asset_dataset(path=path, name=dataset_name, split_cfg=split_cfg)
    split = gather_split_arrays(asset, split_cfg)

    selected_cfg, selected_epoch, val_metrics = select_anfis_config(
        split=split,
        train_cfg=train_cfg,
        tuning_seed=tuning_seed,
        fast=fast,
        fixed_anfis_cfg=fixed_anfis_cfg,
    )

    rows: List[Dict[str, float | int | str]] = []
    metrics_by_seed: Dict[str, Dict[str, object]] = {}
    for seed in eval_seeds:
        metrics = fit_and_eval_anfis(
            split=split,
            anfis_cfg=selected_cfg,
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
    anfis_cfg: ANFISConfig,
    selected_epoch: int,
    sample_counts: Dict[str, int],
    val_metrics: Dict[str, object] | None,
    test_metrics: Dict[str, object],
    scaler: FeatureScaler,
    model: ANFISMultiOutputRegressor,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "anfis_state_dict.pt")
    np.savez(output_path / "anfis_scaler.npz", mean=scaler.scaler.mean_, scale=scaler.scaler.scale_)

    payload = {
        "dataset": dataset_name,
        "model": "anfis_only",
        "target_definition": "model predicts next-day log-return for Open, High, Low, Close",
        "evaluation_definition": "benchmark/train metrics are computed on algebraically reconstructed next-day OHLC prices",
        "target_order": TARGET_KEYS,
        "split_config": asdict(split_cfg),
        "train_config": asdict(train_cfg),
        "selected_config": asdict(anfis_cfg),
        "selected_epoch": int(selected_epoch),
        "sample_counts": sample_counts,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(output_path / "anfis_training_summary.json", payload)
