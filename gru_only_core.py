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

from hybrid_core import FEATURE_COLS, DataConfig, FeatureScaler, TrainConfig, compute_features, save_json, set_seed


DEFAULT_DATASETS: Dict[str, str] = {
    "AMZN": "AMZN.csv",
    "JPM": "JPM.csv",
    "TSLA": "TSLA.csv",
}

PRICE_TARGETS: List[Tuple[str, str]] = [
    ("open", "Open"),
    ("high", "High"),
    ("low", "Low"),
    ("close", "Close"),
]

TARGET_KEYS: List[str] = [key for key, _ in PRICE_TARGETS]
TARGET_COLS: List[str] = [col for _, col in PRICE_TARGETS]
TARGET_DIM = len(TARGET_COLS)


@dataclass
class RatioSplitConfig:
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int | None = None

    @property
    def effective_gap(self) -> int:
        return self.horizon if self.gap is None else self.gap


@dataclass
class GRUConfig:
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 1e-3


@dataclass
class RatioSplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def has_val(self) -> bool:
        return len(self.val) > 0


@dataclass
class OHLCSampleSet:
    asset_name: str
    X: np.ndarray
    y_z: np.ndarray
    target_scale: np.ndarray
    current_ohlc: np.ndarray
    next_ohlc: np.ndarray


@dataclass
class SplitArrays:
    X_train_raw: np.ndarray
    y_train_z: np.ndarray
    scale_train: np.ndarray
    current_ohlc_train: np.ndarray
    next_ohlc_train: np.ndarray
    X_val_raw: np.ndarray
    y_val_z: np.ndarray
    scale_val: np.ndarray
    current_ohlc_val: np.ndarray
    next_ohlc_val: np.ndarray
    X_test_raw: np.ndarray
    y_test_z: np.ndarray
    scale_test: np.ndarray
    current_ohlc_test: np.ndarray
    next_ohlc_test: np.ndarray

    @property
    def has_val(self) -> bool:
        return len(self.y_val_z) > 0

    def sample_counts(self) -> Dict[str, int]:
        return {
            "train": int(len(self.y_train_z)),
            "val": int(len(self.y_val_z)),
            "test": int(len(self.y_test_z)),
        }


class GRUDataset(Dataset):
    def __init__(self, X: np.ndarray, y_z: np.ndarray, scale: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_z, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.scale[idx]


class GRUMultiOutputRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.10,
        output_dim: int = TARGET_DIM,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.direction_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, X_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(X_seq)
        state = out[:, -1]
        pred = self.head(state)
        direction_logit = self.direction_head(state)
        return pred, direction_logit


def parse_ratio(split_ratio: str) -> Tuple[int, ...]:
    normalized = split_ratio.replace(":", "/").replace("-", "/").replace(" ", "")
    if normalized not in {"7/3", "7/2/1"}:
        raise ValueError("split_ratio must be '7/3' or '7/2/1'")
    return tuple(int(part) for part in normalized.split("/"))


def default_gru_candidates() -> List[GRUConfig]:
    return [
        GRUConfig(hidden_dim=24, num_layers=1, dropout=0.10, lr=1e-3),
        GRUConfig(hidden_dim=32, num_layers=1, dropout=0.10, lr=1e-3),
        GRUConfig(hidden_dim=48, num_layers=1, dropout=0.00, lr=1e-3),
        GRUConfig(hidden_dim=32, num_layers=2, dropout=0.10, lr=5e-4),
    ]


def resolve_datasets(
    dataset_names: Sequence[str] | None,
    files: Sequence[str] | None,
) -> List[Tuple[str, Path]]:
    if files:
        resolved: List[Tuple[str, Path]] = []
        for file_path in files:
            path = Path(file_path)
            resolved.append((path.stem.upper(), path))
        return resolved

    requested = [name.upper() for name in dataset_names] if dataset_names else list(DEFAULT_DATASETS)
    resolved = []
    for name in requested:
        if name not in DEFAULT_DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Expected one of {sorted(DEFAULT_DATASETS)}")
        resolved.append((name, Path(DEFAULT_DATASETS[name])))
    return resolved


def _rolling_target_scale(log_prices: np.ndarray, end_idx: int, window: int = 20) -> float:
    recent_returns = np.diff(log_prices[max(0, end_idx - window) : end_idx + 1])
    scale = float(np.nanstd(recent_returns))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = 1e-4
    return scale


def build_ohlc_return_samples(
    df,
    asset_name: str,
    split_cfg: RatioSplitConfig,
) -> OHLCSampleSet:
    feature_matrix = df[FEATURE_COLS].values.astype(np.float32)
    price_matrix = df[TARGET_COLS].values.astype(np.float32)
    log_prices = {col: np.log(df[col].values.astype(np.float32)) for col in TARGET_COLS}

    xs, ys, scales = [], [], []
    current_prices, next_prices = [], []

    for end_idx in range(split_cfg.seq_len - 1, len(df) - split_cfg.horizon):
        window = feature_matrix[end_idx - split_cfg.seq_len + 1 : end_idx + 1]
        if np.isnan(window).any():
            continue

        raw_targets = []
        target_scales = []
        for col in TARGET_COLS:
            series = log_prices[col]
            raw_target = float(series[end_idx + split_cfg.horizon] - series[end_idx])
            scale = _rolling_target_scale(series, end_idx=end_idx)
            raw_targets.append(raw_target)
            target_scales.append(scale)

        raw_targets_arr = np.array(raw_targets, dtype=np.float32)
        target_scales_arr = np.array(target_scales, dtype=np.float32)

        xs.append(window)
        ys.append(raw_targets_arr / target_scales_arr)
        scales.append(target_scales_arr)
        current_prices.append(price_matrix[end_idx])
        next_prices.append(price_matrix[end_idx + split_cfg.horizon])

    return OHLCSampleSet(
        asset_name=asset_name,
        X=np.stack(xs).astype(np.float32),
        y_z=np.stack(ys).astype(np.float32),
        target_scale=np.stack(scales).astype(np.float32),
        current_ohlc=np.stack(current_prices).astype(np.float32),
        next_ohlc=np.stack(next_prices).astype(np.float32),
    )


def load_asset_dataset(path: str | Path, name: str, split_cfg: RatioSplitConfig) -> OHLCSampleSet:
    import pandas as pd

    raw = pd.read_csv(path)
    featured = compute_features(raw)
    return build_ohlc_return_samples(featured, asset_name=name, split_cfg=split_cfg)


def make_purged_ratio_split(n_samples: int, split_cfg: RatioSplitConfig) -> RatioSplitIndices:
    parts = parse_ratio(split_cfg.split_ratio)
    gap = split_cfg.effective_gap
    n_gaps = len(parts) - 1
    usable = n_samples - n_gaps * gap
    if usable < len(parts):
        raise ValueError(
            f"Not enough samples for split {split_cfg.split_ratio} with gap={gap}. "
            f"Need more than {n_gaps * gap + len(parts)} samples, got {n_samples}."
        )

    block_sizes: List[int] = []
    remaining = usable
    for idx, weight in enumerate(parts):
        if idx == len(parts) - 1:
            size = remaining
        else:
            remaining_weight = sum(parts[idx:])
            min_tail = len(parts) - idx - 1
            size = max(1, int(round(remaining * weight / remaining_weight)))
            size = min(size, remaining - min_tail)
        block_sizes.append(size)
        remaining -= size

    train_size = block_sizes[0]
    cursor = train_size
    train = np.arange(0, train_size, dtype=np.int64)

    if len(parts) == 2:
        cursor += gap
        test = np.arange(cursor, cursor + block_sizes[1], dtype=np.int64)
        val = np.array([], dtype=np.int64)
    else:
        cursor += gap
        val = np.arange(cursor, cursor + block_sizes[1], dtype=np.int64)
        cursor = cursor + block_sizes[1] + gap
        test = np.arange(cursor, cursor + block_sizes[2], dtype=np.int64)

    return RatioSplitIndices(train=train, val=val, test=test)


def gather_split_arrays(asset: OHLCSampleSet, split_cfg: RatioSplitConfig) -> SplitArrays:
    indices = make_purged_ratio_split(len(asset.X), split_cfg)
    return SplitArrays(
        X_train_raw=asset.X[indices.train],
        y_train_z=asset.y_z[indices.train],
        scale_train=asset.target_scale[indices.train],
        current_ohlc_train=asset.current_ohlc[indices.train],
        next_ohlc_train=asset.next_ohlc[indices.train],
        X_val_raw=asset.X[indices.val],
        y_val_z=asset.y_z[indices.val],
        scale_val=asset.target_scale[indices.val],
        current_ohlc_val=asset.current_ohlc[indices.val],
        next_ohlc_val=asset.next_ohlc[indices.val],
        X_test_raw=asset.X[indices.test],
        y_test_z=asset.y_z[indices.test],
        scale_test=asset.target_scale[indices.test],
        current_ohlc_test=asset.current_ohlc[indices.test],
        next_ohlc_test=asset.next_ohlc[indices.test],
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def sign_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _scaled_predictions(y_z: np.ndarray, pred_z: np.ndarray, scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return y_z * scale, pred_z * scale


def reconstruct_prices_from_logreturns(current_ohlc: np.ndarray, pred_logreturns: np.ndarray) -> np.ndarray:
    current = np.maximum(np.asarray(current_ohlc, dtype=np.float32), 1e-8)
    pred = np.asarray(pred_logreturns, dtype=np.float32)
    return current * np.exp(pred)


def _metrics_from_price_arrays(
    current_ohlc: np.ndarray,
    actual_next_ohlc: np.ndarray,
    pred_next_ohlc: np.ndarray,
) -> Dict[str, object]:
    by_target: Dict[str, Dict[str, float]] = {}
    for idx, target_key in enumerate(TARGET_KEYS):
        y_col = actual_next_ohlc[:, idx]
        pred_col = pred_next_ohlc[:, idx]
        current_col = current_ohlc[:, idx]
        by_target[target_key] = {
            "rmse": rmse(y_col, pred_col),
            "mape": mape(y_col, pred_col),
            "mae": mae(y_col, pred_col),
            "r2": r2_score(y_col, pred_col),
            "sign_acc": sign_acc(y_col - current_col, pred_col - current_col),
        }

    selection_score = float(np.mean([metrics["rmse"] for metrics in by_target.values()]))
    return {"by_target": by_target, "selection_score": selection_score}


def predict_gru(
    model: GRUMultiOutputRegressor,
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


def evaluate_gru(
    model: GRUMultiOutputRegressor,
    dataset: GRUDataset,
    current_ohlc: np.ndarray,
    actual_next_ohlc: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, object]:
    pred_z, y_z, scale = predict_gru(model, dataset, batch_size=batch_size)
    _y_true, y_pred = _scaled_predictions(y_z, pred_z, scale)
    pred_next_ohlc = reconstruct_prices_from_logreturns(current_ohlc, y_pred)
    return _metrics_from_price_arrays(current_ohlc, actual_next_ohlc, pred_next_ohlc)


def _make_model(input_dim: int, gru_cfg: GRUConfig) -> GRUMultiOutputRegressor:
    return GRUMultiOutputRegressor(
        input_dim=input_dim,
        hidden_dim=gru_cfg.hidden_dim,
        num_layers=gru_cfg.num_layers,
        dropout=gru_cfg.dropout,
        output_dim=TARGET_DIM,
    )


def train_gru_with_early_stopping(
    model: GRUMultiOutputRegressor,
    train_ds: GRUDataset,
    val_ds: GRUDataset,
    val_current_ohlc: np.ndarray,
    val_next_ohlc: np.ndarray,
    gru_cfg: GRUConfig,
    train_cfg: TrainConfig,
) -> Tuple[GRUMultiOutputRegressor, Dict[str, object], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=gru_cfg.lr, weight_decay=train_cfg.weight_decay)
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

        metrics = evaluate_gru(
            model,
            val_ds,
            current_ohlc=val_current_ohlc,
            actual_next_ohlc=val_next_ohlc,
            batch_size=train_cfg.batch_size,
        )
        if metrics["selection_score"] < best_score - 1e-5:
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
    final_metrics = evaluate_gru(
        model,
        val_ds,
        current_ohlc=val_current_ohlc,
        actual_next_ohlc=val_next_ohlc,
        batch_size=train_cfg.batch_size,
    )
    return model, final_metrics, best_epoch


def fit_gru_fixed_epochs(
    model: GRUMultiOutputRegressor,
    train_ds: GRUDataset,
    gru_cfg: GRUConfig,
    train_cfg: TrainConfig,
    epochs: int,
) -> GRUMultiOutputRegressor:
    optimizer = torch.optim.AdamW(model.parameters(), lr=gru_cfg.lr, weight_decay=train_cfg.weight_decay)
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


def tune_gru_config(
    split: SplitArrays,
    train_cfg: TrainConfig,
    candidates: Sequence[GRUConfig],
    seed: int,
) -> Tuple[GRUConfig, int, Dict[str, object]]:
    if not split.has_val:
        raise ValueError("Validation split is required to tune GRU configs.")

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
        model = _make_model(input_dim=X_train.shape[-1], gru_cfg=cfg)
        _, metrics, best_epoch_here = train_gru_with_early_stopping(
            model,
            train_ds,
            val_ds,
            val_current_ohlc=split.current_ohlc_val,
            val_next_ohlc=split.next_ohlc_val,
            gru_cfg=cfg,
            train_cfg=train_cfg,
        )
        if float(metrics["selection_score"]) < float(best_metrics["selection_score"]) - 1e-5:
            best_cfg = cfg
            best_epoch = best_epoch_here
            best_metrics = metrics

    return best_cfg, best_epoch, best_metrics


def fit_and_eval_gru(
    split: SplitArrays,
    gru_cfg: GRUConfig,
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
    model = _make_model(input_dim=X_train.shape[-1], gru_cfg=gru_cfg)
    model = fit_gru_fixed_epochs(model, train_ds, gru_cfg, train_cfg, epochs=epochs or train_cfg.max_epochs)
    return evaluate_gru(
        model,
        test_ds,
        current_ohlc=split.current_ohlc_test,
        actual_next_ohlc=split.next_ohlc_test,
        batch_size=train_cfg.batch_size,
    )


def fit_final_gru(
    split: SplitArrays,
    gru_cfg: GRUConfig,
    train_cfg: TrainConfig,
    seed: int,
    epochs: int | None = None,
) -> Tuple[GRUMultiOutputRegressor, FeatureScaler, Dict[str, object]]:
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
    model = _make_model(input_dim=X_train.shape[-1], gru_cfg=gru_cfg)
    model = fit_gru_fixed_epochs(model, train_ds, gru_cfg, train_cfg, epochs=epochs or train_cfg.max_epochs)
    test_metrics = evaluate_gru(
        model,
        test_ds,
        current_ohlc=split.current_ohlc_test,
        actual_next_ohlc=split.next_ohlc_test,
        batch_size=train_cfg.batch_size,
    )
    return model, scaler, test_metrics


def select_gru_config(
    split: SplitArrays,
    train_cfg: TrainConfig,
    tuning_seed: int,
    fast: bool = False,
    fixed_gru_cfg: GRUConfig | None = None,
) -> Tuple[GRUConfig, int, Dict[str, object] | None]:
    if split.has_val:
        if fixed_gru_cfg is not None:
            return fixed_gru_cfg, train_cfg.max_epochs, None

        candidates = default_gru_candidates()
        if fast:
            candidates = candidates[:2]
        selected_cfg, selected_epoch, val_metrics = tune_gru_config(
            split=split,
            train_cfg=train_cfg,
            candidates=candidates,
            seed=tuning_seed,
        )
        return selected_cfg, selected_epoch, val_metrics

    if fixed_gru_cfg is None:
        raise ValueError(
            "Split 7/3 has no validation set. To keep the protocol leak-free, "
            "you must pass an explicit fixed GRU config."
        )
    return fixed_gru_cfg, train_cfg.max_epochs, None


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
                "model": "gru_only",
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
    fixed_gru_cfg: GRUConfig | None = None,
) -> Dict[str, object]:
    asset = load_asset_dataset(path=path, name=dataset_name, split_cfg=split_cfg)
    split = gather_split_arrays(asset, split_cfg)

    selected_cfg, selected_epoch, val_metrics = select_gru_config(
        split=split,
        train_cfg=train_cfg,
        tuning_seed=tuning_seed,
        fast=fast,
        fixed_gru_cfg=fixed_gru_cfg,
    )

    rows: List[Dict[str, float | int | str]] = []
    metrics_by_seed: Dict[str, Dict[str, object]] = {}
    for seed in eval_seeds:
        metrics = fit_and_eval_gru(
            split=split,
            gru_cfg=selected_cfg,
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
    gru_cfg: GRUConfig,
    selected_epoch: int,
    sample_counts: Dict[str, int],
    val_metrics: Dict[str, object] | None,
    test_metrics: Dict[str, object],
    scaler: FeatureScaler,
    model: GRUMultiOutputRegressor,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "gru_state_dict.pt")
    np.savez(output_path / "gru_scaler.npz", mean=scaler.scaler.mean_, scale=scaler.scaler.scale_)

    payload = {
        "dataset": dataset_name,
        "model": "gru_only",
        "target_definition": "model predicts next-day log-return for Open, High, Low, Close",
        "evaluation_definition": "benchmark/train metrics are computed on algebraically reconstructed next-day OHLC prices",
        "target_order": TARGET_KEYS,
        "split_config": asdict(split_cfg),
        "train_config": asdict(train_cfg),
        "selected_config": asdict(gru_cfg),
        "selected_epoch": int(selected_epoch),
        "sample_counts": sample_counts,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(output_path / "gru_training_summary.json", payload)
