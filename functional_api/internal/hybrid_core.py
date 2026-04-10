from __future__ import annotations

import copy
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


FEATURE_COLS: List[str] = [
    "ret1", "gap", "body", "range", "vol_chg",
    "mom_3", "mom_5", "mom_10", "mom_20", "mom_60",
    "vol_3", "vol_5", "vol_10", "vol_20", "vol_60",
    "close_sma_3", "close_sma_5", "close_sma_10", "close_sma_20", "close_sma_60",
    "volu_sma_3", "volu_sma_5", "volu_sma_10", "volu_sma_20", "volu_sma_60",
    "rsi14", "rsi28", "dow_sin", "dow_cos", "mon_sin", "mon_cos",
]

REGIME_FEATURES: List[str] = [
    "mom_5", "mom_20", "mom_60",
    "vol_5", "vol_20", "vol_60",
    "range", "vol_chg", "rsi14", "rsi28",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true)
    mask = denom > eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot <= eps:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def sign_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


@dataclass
class DataConfig:
    seq_len: int = 48
    horizon: int = 20
    n_splits: int = 3
    val_frac: float = 0.10
    test_frac: float = 0.10
    min_train_frac: float = 0.40
    gap: Optional[int] = None
    max_train_size: Optional[int] = 1024

    @property
    def effective_gap(self) -> int:
        return self.horizon if self.gap is None else self.gap


@dataclass
class TrainConfig:
    max_epochs: int = 12
    patience: int = 4
    batch_size: int = 256
    weight_decay: float = 1e-4
    direction_loss_weight: float = 0.25
    entropy_bonus: float = 0.001


@dataclass
class LSTMConfig:
    hidden_dim: int = 24
    dropout: float = 0.10
    lr: float = 1e-3


@dataclass
class HybridConfig:
    channels: int = 20
    summary_hidden: int = 48
    expert_hidden: int = 48
    n_rules: int = 4
    dropout: float = 0.10
    lr: float = 1e-3


@dataclass
class FoldIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class AssetSampleSet:
    asset_name: str
    X: np.ndarray
    y_z: np.ndarray
    target_scale: np.ndarray
    asset_id: np.ndarray


@dataclass
class FoldArrays:
    X_train_raw: np.ndarray
    y_train_z: np.ndarray
    scale_train: np.ndarray
    asset_train: np.ndarray
    X_val_raw: np.ndarray
    y_val_z: np.ndarray
    scale_val: np.ndarray
    asset_val: np.ndarray
    X_test_raw: np.ndarray
    y_test_z: np.ndarray
    scale_test: np.ndarray
    asset_test: np.ndarray


class FeatureScaler:
    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        shape = X.shape
        return self.scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan)
    log_close = np.log(close)

    df["ret1"] = log_close.diff()
    df["gap"] = np.log(open_ / close.shift(1))
    df["body"] = np.log(close / open_)
    df["range"] = np.log(high / low)
    df["vol_chg"] = np.log(volume).diff()

    for window in [3, 5, 10, 20, 60]:
        df[f"mom_{window}"] = log_close - log_close.shift(window)
        df[f"vol_{window}"] = df["ret1"].rolling(window).std()
        df[f"close_sma_{window}"] = close / close.rolling(window).mean() - 1
        df[f"volu_sma_{window}"] = volume / volume.rolling(window).mean() - 1

    delta = close.diff()
    for window, name in [(14, "rsi14"), (28, "rsi28")]:
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        df[name] = 100 - 100 / (1 + rs)

    dow = df["Date"].dt.dayofweek
    month = df["Date"].dt.month
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    df["mon_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df["mon_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    return df


def build_asset_samples(
    df: pd.DataFrame,
    asset_name: str,
    asset_id: int,
    data_cfg: DataConfig,
) -> AssetSampleSet:
    feature_matrix = df[FEATURE_COLS].values.astype(np.float32)
    log_close = np.log(df["Close"].values.astype(np.float32))
    ret1 = df["ret1"].values.astype(np.float32)

    xs, ys, scales, asset_ids = [], [], [], []

    for end_idx in range(data_cfg.seq_len - 1, len(df) - data_cfg.horizon):
        window = feature_matrix[end_idx - data_cfg.seq_len + 1 : end_idx + 1]
        if np.isnan(window).any():
            continue

        raw_target = log_close[end_idx + data_cfg.horizon] - log_close[end_idx]
        scale = np.nanstd(ret1[max(0, end_idx - 20) : end_idx + 1])
        if not np.isfinite(scale) or scale < 1e-4:
            scale = 1e-4

        xs.append(window)
        ys.append(raw_target / scale)
        scales.append(scale)
        asset_ids.append(asset_id)

    return AssetSampleSet(
        asset_name=asset_name,
        X=np.stack(xs).astype(np.float32),
        y_z=np.array(ys, dtype=np.float32),
        target_scale=np.array(scales, dtype=np.float32),
        asset_id=np.array(asset_ids, dtype=np.int64),
    )


def make_purged_walk_forward_splits(n_samples: int, data_cfg: DataConfig) -> List[FoldIndices]:
    gap = data_cfg.effective_gap
    test_size = max(1, int(round(n_samples * data_cfg.test_frac)))
    val_size = max(1, int(round(n_samples * data_cfg.val_frac)))

    first_test_start = max(
        int(round(n_samples * data_cfg.min_train_frac)),
        n_samples - data_cfg.n_splits * test_size,
    )

    folds: List[FoldIndices] = []
    for fold_idx in range(data_cfg.n_splits):
        test_start = first_test_start + fold_idx * test_size
        test_end = min(test_start + test_size, n_samples)

        val_end = test_start - gap
        val_start = max(0, val_end - val_size)

        train_end = val_start - gap
        if data_cfg.max_train_size is None:
            train_start = 0
        else:
            train_start = max(0, train_end - data_cfg.max_train_size)

        if train_end <= train_start or val_start >= val_end or test_start >= test_end:
            continue

        folds.append(
            FoldIndices(
                train=np.arange(train_start, train_end),
                val=np.arange(val_start, val_end),
                test=np.arange(test_start, test_end),
            )
        )
    return folds


def load_all_assets(files: Sequence[Tuple[str, str]], data_cfg: DataConfig) -> List[AssetSampleSet]:
    assets: List[AssetSampleSet] = []
    for asset_id, (path, name) in enumerate(files):
        raw = pd.read_csv(path)
        featured = compute_features(raw)
        assets.append(build_asset_samples(featured, name, asset_id, data_cfg))
    return assets


def gather_fold_arrays(assets: Sequence[AssetSampleSet], data_cfg: DataConfig, fold_idx: int) -> FoldArrays:
    per_asset_splits = [make_purged_walk_forward_splits(len(asset.X), data_cfg)[fold_idx] for asset in assets]

    def concat(key: str, split_name: str) -> np.ndarray:
        arrays: List[np.ndarray] = []
        for asset, split in zip(assets, per_asset_splits):
            indices = getattr(split, split_name)
            arrays.append(getattr(asset, key)[indices])
        return np.concatenate(arrays, axis=0)

    return FoldArrays(
        X_train_raw=concat("X", "train"),
        y_train_z=concat("y_z", "train"),
        scale_train=concat("target_scale", "train"),
        asset_train=concat("asset_id", "train"),
        X_val_raw=concat("X", "val"),
        y_val_z=concat("y_z", "val"),
        scale_val=concat("target_scale", "val"),
        asset_val=concat("asset_id", "val"),
        X_test_raw=concat("X", "test"),
        y_test_z=concat("y_z", "test"),
        scale_test=concat("target_scale", "test"),
        asset_test=concat("asset_id", "test"),
    )


def fit_scaler_and_transform(*arrays: np.ndarray, fit_on: np.ndarray) -> Tuple[FeatureScaler, List[np.ndarray]]:
    scaler = FeatureScaler().fit(fit_on)
    return scaler, [scaler.transform(arr) for arr in arrays]


def summarize_windows(X: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            X[:, -1, :],
            X[:, -5:, :].mean(axis=1),
            X[:, -20:, :].mean(axis=1),
            X[:, -20:, :].std(axis=1),
            X.mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def window_features_for_linear(X: np.ndarray, asset_id: np.ndarray, n_assets: int) -> np.ndarray:
    one_hot = np.eye(n_assets, dtype=np.float32)[asset_id]
    return np.concatenate([X.reshape(len(X), -1), one_hot], axis=1)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y_z: np.ndarray, asset_id: np.ndarray, scale: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_z, dtype=torch.float32)
        self.asset_id = torch.tensor(asset_id, dtype=torch.long)
        self.scale = torch.tensor(scale, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.asset_id[idx], self.y[idx], self.scale[idx]


class HybridDataset(Dataset):
    def __init__(
        self,
        X_seq: np.ndarray,
        X_summary: np.ndarray,
        y_z: np.ndarray,
        asset_id: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_summary = torch.tensor(X_summary, dtype=torch.float32)
        self.y = torch.tensor(y_z, dtype=torch.float32)
        self.asset_id = torch.tensor(asset_id, dtype=torch.long)
        self.scale = torch.tensor(scale, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.X_seq[idx],
            self.X_summary[idx],
            self.asset_id[idx],
            self.y[idx],
            self.scale[idx],
        )


class LSTMDirectionalRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        hidden_dim: int = 24,
        emb_dim: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, emb_dim)
        self.lstm = nn.LSTM(input_dim + emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.direction_head = nn.Linear(hidden_dim, 1)

    def forward(self, X_seq: torch.Tensor, asset_id: torch.Tensor):
        emb = self.asset_emb(asset_id).unsqueeze(1).expand(-1, X_seq.size(1), -1)
        out, _ = self.lstm(torch.cat([X_seq, emb], dim=-1))
        state = out[:, -1]
        pred = self.head(state).squeeze(-1)
        direction_logit = self.direction_head(state).squeeze(-1)
        return pred, direction_logit


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.10) -> None:
        super().__init__()
        padding = (3 - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.group_norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)[:, :, : x.size(-1)]
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)[:, :, : x.size(-1)]
        y = self.group_norm(y)
        return F.gelu(x + y)


class RegimeAwareFuzzyHybrid(nn.Module):
    def __init__(
        self,
        seq_input_dim: int,
        summary_input_dim: int,
        n_assets: int,
        channels: int = 20,
        summary_hidden: int = 48,
        expert_hidden: int = 48,
        n_rules: int = 4,
        emb_dim: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, emb_dim)
        self.regime_idx = [FEATURE_COLS.index(col) for col in REGIME_FEATURES]

        self.sequence_projection = nn.Linear(seq_input_dim + emb_dim, channels)
        self.tcn_blocks = nn.ModuleList(
            [ResidualTCNBlock(channels=channels, dilation=2**i, dropout=dropout) for i in range(3)]
        )

        self.summary_encoder = nn.Sequential(
            nn.LayerNorm(summary_input_dim + emb_dim),
            nn.Linear(summary_input_dim + emb_dim, summary_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        gate_dim = len(self.regime_idx) + emb_dim + 2
        self.centers = nn.Parameter(torch.randn(n_rules, gate_dim) * 0.2)
        self.log_scales = nn.Parameter(torch.zeros(n_rules, gate_dim))
        self.rule_bias = nn.Parameter(torch.zeros(n_rules))
        self.gate_mlp = nn.Sequential(nn.LayerNorm(gate_dim), nn.Linear(gate_dim, n_rules))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        expert_input_dim = channels + summary_hidden + emb_dim
        self.linear_shortcut = nn.Sequential(
            nn.Linear(summary_input_dim + emb_dim, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(expert_input_dim),
                    nn.Linear(expert_input_dim, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, 1),
                )
                for _ in range(n_rules)
            ]
        )
        self.direction_head = nn.Linear(expert_input_dim, 1)

    def fuzzy_logits(self, z: torch.Tensor) -> torch.Tensor:
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)
        scales = F.softplus(self.log_scales).unsqueeze(0) + 1e-3
        return -0.5 * ((diff / scales) ** 2).sum(dim=-1) + self.rule_bias

    def forward(self, X_seq: torch.Tensor, X_summary: torch.Tensor, asset_id: torch.Tensor):
        emb = self.asset_emb(asset_id)

        emb_seq = emb.unsqueeze(1).expand(-1, X_seq.size(1), -1)
        seq_hidden = self.sequence_projection(torch.cat([X_seq, emb_seq], dim=-1)).transpose(1, 2)
        for block in self.tcn_blocks:
            seq_hidden = block(seq_hidden)
        seq_state = seq_hidden[:, :, -1]

        summary_state = self.summary_encoder(torch.cat([X_summary, emb], dim=-1))

        regime_features = X_seq[:, -1, self.regime_idx]
        ret_mean_5 = X_seq[:, -5:, FEATURE_COLS.index("ret1")].mean(dim=1)
        ret_std_20 = X_seq[:, -20:, FEATURE_COLS.index("ret1")].std(dim=1)
        gate_input = torch.cat([regime_features, emb, ret_mean_5.unsqueeze(-1), ret_std_20.unsqueeze(-1)], dim=-1)

        gate = F.softmax(
            (self.fuzzy_logits(gate_input) + self.gate_mlp(gate_input))
            / (F.softplus(self.temperature) + 1e-3),
            dim=-1,
        )

        expert_input = torch.cat([seq_state, summary_state, emb], dim=-1)
        expert_outputs = torch.stack([expert(expert_input).squeeze(-1) for expert in self.experts], dim=-1)

        pred = self.linear_shortcut(torch.cat([X_summary, emb], dim=-1)).squeeze(-1)
        pred = pred + (gate * expert_outputs).sum(dim=-1)
        direction_logit = self.direction_head(expert_input).squeeze(-1)
        return pred, direction_logit, gate


def _eval_scaled_predictions(y_z: np.ndarray, pred_z: np.ndarray, scale: np.ndarray) -> Dict[str, float]:
    y = y_z * scale
    pred = pred_z * scale
    return {
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "mape": mape(y, pred),
        "r2": r2(y, pred),
        "sign_acc": sign_acc(y, pred),
    }


def predict_lstm(
    model: nn.Module,
    dataset: SequenceDataset,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    preds, ys, scales = [], [], []
    with torch.no_grad():
        for X_seq, asset_id, y_z, scale in loader:
            pred_z, _ = model(X_seq, asset_id)
            preds.append(pred_z.cpu().numpy())
            ys.append(y_z.cpu().numpy())
            scales.append(scale.cpu().numpy())

    return np.concatenate(preds), np.concatenate(ys), np.concatenate(scales)


def predict_hybrid(
    model: nn.Module,
    dataset: HybridDataset,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    preds, ys, scales = [], [], []
    with torch.no_grad():
        for X_seq, X_summary, asset_id, y_z, scale in loader:
            pred_z, _, _ = model(X_seq, X_summary, asset_id)
            preds.append(pred_z.cpu().numpy())
            ys.append(y_z.cpu().numpy())
            scales.append(scale.cpu().numpy())

    return np.concatenate(preds), np.concatenate(ys), np.concatenate(scales)


def train_lstm_with_early_stopping(
    model: LSTMDirectionalRegressor,
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    lstm_cfg: LSTMConfig,
    train_cfg: TrainConfig,
) -> Tuple[LSTMDirectionalRegressor, Dict[str, float], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lstm_cfg.lr, weight_decay=train_cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    best_score = float("inf")
    best_state = None
    best_epoch = 1
    bad_epochs = 0

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        for X_seq, asset_id, y_z, _scale in train_loader:
            pred_z, direction_logit = model(X_seq, asset_id)
            loss = F.smooth_l1_loss(pred_z, y_z)
            loss = loss + train_cfg.direction_loss_weight * F.binary_cross_entropy_with_logits(
                direction_logit, (y_z > 0).float()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        pred_z, y_z, scale = predict_lstm(model, val_ds, batch_size=train_cfg.batch_size)
        metrics = _eval_scaled_predictions(y_z, pred_z, scale)
        score = metrics["rmse"]

        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_pred_z, final_y_z, final_scale = predict_lstm(model, val_ds, batch_size=train_cfg.batch_size)
    return model, _eval_scaled_predictions(final_y_z, final_pred_z, final_scale), best_epoch


def train_hybrid_with_early_stopping(
    model: RegimeAwareFuzzyHybrid,
    train_ds: HybridDataset,
    val_ds: HybridDataset,
    hybrid_cfg: HybridConfig,
    train_cfg: TrainConfig,
) -> Tuple[RegimeAwareFuzzyHybrid, Dict[str, float], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=hybrid_cfg.lr, weight_decay=train_cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    best_score = float("inf")
    best_state = None
    best_epoch = 1
    bad_epochs = 0

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        for X_seq, X_summary, asset_id, y_z, _scale in train_loader:
            pred_z, direction_logit, gate = model(X_seq, X_summary, asset_id)
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

        pred_z, y_z, scale = predict_hybrid(model, val_ds, batch_size=train_cfg.batch_size)
        metrics = _eval_scaled_predictions(y_z, pred_z, scale)
        score = metrics["rmse"]

        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_pred_z, final_y_z, final_scale = predict_hybrid(model, val_ds, batch_size=train_cfg.batch_size)
    return model, _eval_scaled_predictions(final_y_z, final_pred_z, final_scale), best_epoch


def fit_lstm_fixed_epochs(
    model: LSTMDirectionalRegressor,
    train_ds: SequenceDataset,
    lstm_cfg: LSTMConfig,
    train_cfg: TrainConfig,
    epochs: int,
) -> LSTMDirectionalRegressor:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lstm_cfg.lr, weight_decay=train_cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    for _ in range(max(1, epochs)):
        model.train()
        for X_seq, asset_id, y_z, _scale in train_loader:
            pred_z, direction_logit = model(X_seq, asset_id)
            loss = F.smooth_l1_loss(pred_z, y_z)
            loss = loss + train_cfg.direction_loss_weight * F.binary_cross_entropy_with_logits(
                direction_logit, (y_z > 0).float()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def fit_hybrid_fixed_epochs(
    model: RegimeAwareFuzzyHybrid,
    train_ds: HybridDataset,
    hybrid_cfg: HybridConfig,
    train_cfg: TrainConfig,
    epochs: int,
) -> RegimeAwareFuzzyHybrid:
    optimizer = torch.optim.AdamW(model.parameters(), lr=hybrid_cfg.lr, weight_decay=train_cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    for _ in range(max(1, epochs)):
        model.train()
        for X_seq, X_summary, asset_id, y_z, _scale in train_loader:
            pred_z, direction_logit, gate = model(X_seq, X_summary, asset_id)
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


def evaluate_lstm(model: LSTMDirectionalRegressor, dataset: SequenceDataset, batch_size: int = 256) -> Dict[str, float]:
    pred_z, y_z, scale = predict_lstm(model, dataset, batch_size=batch_size)
    return _eval_scaled_predictions(y_z, pred_z, scale)


def evaluate_hybrid(
    model: RegimeAwareFuzzyHybrid, dataset: HybridDataset, batch_size: int = 256
) -> Dict[str, float]:
    pred_z, y_z, scale = predict_hybrid(model, dataset, batch_size=batch_size)
    return _eval_scaled_predictions(y_z, pred_z, scale)


def tune_ridge_alpha(
    fold: FoldArrays,
    n_assets: int,
    alphas: Sequence[float],
) -> Tuple[float, Dict[str, float]]:
    scaler = FeatureScaler().fit(fold.X_train_raw)
    X_train = scaler.transform(fold.X_train_raw)
    X_val = scaler.transform(fold.X_val_raw)

    train_features = window_features_for_linear(X_train, fold.asset_train, n_assets)
    val_features = window_features_for_linear(X_val, fold.asset_val, n_assets)

    best_alpha = float(alphas[0])
    best_metrics = {"rmse": float("inf"), "mae": float("inf"), "sign_acc": 0.0}

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(train_features, fold.y_train_z)
        pred_z = model.predict(val_features)
        metrics = _eval_scaled_predictions(fold.y_val_z, pred_z, fold.scale_val)
        if metrics["rmse"] < best_metrics["rmse"] - 1e-5:
            best_alpha = float(alpha)
            best_metrics = metrics
    return best_alpha, best_metrics


def fit_and_eval_ridge(
    fold: FoldArrays,
    n_assets: int,
    alpha: float,
) -> Dict[str, float]:
    X_trainval_raw = np.concatenate([fold.X_train_raw, fold.X_val_raw], axis=0)
    y_trainval_z = np.concatenate([fold.y_train_z, fold.y_val_z], axis=0)
    asset_trainval = np.concatenate([fold.asset_train, fold.asset_val], axis=0)

    scaler = FeatureScaler().fit(X_trainval_raw)
    X_trainval = scaler.transform(X_trainval_raw)
    X_test = scaler.transform(fold.X_test_raw)

    model = Ridge(alpha=alpha)
    model.fit(window_features_for_linear(X_trainval, asset_trainval, n_assets), y_trainval_z)
    pred_z = model.predict(window_features_for_linear(X_test, fold.asset_test, n_assets))
    return _eval_scaled_predictions(fold.y_test_z, pred_z, fold.scale_test)


def default_lstm_candidates() -> List[LSTMConfig]:
    return [
        LSTMConfig(hidden_dim=24, dropout=0.10, lr=1e-3),
        LSTMConfig(hidden_dim=32, dropout=0.10, lr=1e-3),
        LSTMConfig(hidden_dim=48, dropout=0.00, lr=1e-3),
        LSTMConfig(hidden_dim=32, dropout=0.10, lr=5e-4),
    ]


def default_hybrid_candidates() -> List[HybridConfig]:
    return [
        HybridConfig(channels=20, summary_hidden=48, expert_hidden=48, n_rules=4, dropout=0.10, lr=1e-3),
        HybridConfig(channels=24, summary_hidden=48, expert_hidden=48, n_rules=4, dropout=0.10, lr=1e-3),
        HybridConfig(channels=20, summary_hidden=64, expert_hidden=64, n_rules=4, dropout=0.10, lr=1e-3),
        HybridConfig(channels=20, summary_hidden=48, expert_hidden=48, n_rules=6, dropout=0.10, lr=5e-4),
    ]


def _scaled_views_for_tuning(
    fold: FoldArrays,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = FeatureScaler().fit(fold.X_train_raw)
    X_train = scaler.transform(fold.X_train_raw)
    X_val = scaler.transform(fold.X_val_raw)
    X_test = scaler.transform(fold.X_test_raw)
    return X_train, X_val, X_test


def tune_lstm_config(
    fold: FoldArrays,
    n_assets: int,
    train_cfg: TrainConfig,
    candidates: Sequence[LSTMConfig],
    seed: int,
) -> Tuple[LSTMConfig, int, Dict[str, float]]:
    X_train, X_val, _X_test = _scaled_views_for_tuning(fold)
    train_ds = SequenceDataset(X_train, fold.y_train_z, fold.asset_train, fold.scale_train)
    val_ds = SequenceDataset(X_val, fold.y_val_z, fold.asset_val, fold.scale_val)

    best_cfg = candidates[0]
    best_epoch = 1
    best_metrics = {"rmse": float("inf"), "mae": float("inf"), "sign_acc": 0.0}

    for cfg in candidates:
        set_seed(seed)
        model = LSTMDirectionalRegressor(
            input_dim=len(FEATURE_COLS),
            n_assets=n_assets,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )
        _, metrics, best_epoch_here = train_lstm_with_early_stopping(model, train_ds, val_ds, cfg, train_cfg)
        if metrics["rmse"] < best_metrics["rmse"] - 1e-5:
            best_cfg = cfg
            best_epoch = best_epoch_here
            best_metrics = metrics
    return best_cfg, best_epoch, best_metrics


def fit_and_eval_lstm(
    fold: FoldArrays,
    n_assets: int,
    cfg: LSTMConfig,
    train_cfg: TrainConfig,
    best_epoch: int,
    seed: int,
) -> Dict[str, float]:
    X_trainval_raw = np.concatenate([fold.X_train_raw, fold.X_val_raw], axis=0)
    y_trainval_z = np.concatenate([fold.y_train_z, fold.y_val_z], axis=0)
    asset_trainval = np.concatenate([fold.asset_train, fold.asset_val], axis=0)
    scale_trainval = np.concatenate([fold.scale_train, fold.scale_val], axis=0)

    scaler = FeatureScaler().fit(X_trainval_raw)
    X_trainval = scaler.transform(X_trainval_raw)
    X_test = scaler.transform(fold.X_test_raw)

    train_ds = SequenceDataset(X_trainval, y_trainval_z, asset_trainval, scale_trainval)
    test_ds = SequenceDataset(X_test, fold.y_test_z, fold.asset_test, fold.scale_test)

    set_seed(seed)
    model = LSTMDirectionalRegressor(
        input_dim=len(FEATURE_COLS),
        n_assets=n_assets,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    )
    model = fit_lstm_fixed_epochs(model, train_ds, cfg, train_cfg, epochs=best_epoch)
    return evaluate_lstm(model, test_ds, batch_size=train_cfg.batch_size)


def tune_hybrid_config(
    fold: FoldArrays,
    n_assets: int,
    train_cfg: TrainConfig,
    candidates: Sequence[HybridConfig],
    seed: int,
) -> Tuple[HybridConfig, int, Dict[str, float]]:
    X_train, X_val, _X_test = _scaled_views_for_tuning(fold)
    S_train = summarize_windows(X_train)
    S_val = summarize_windows(X_val)

    train_ds = HybridDataset(X_train, S_train, fold.y_train_z, fold.asset_train, fold.scale_train)
    val_ds = HybridDataset(X_val, S_val, fold.y_val_z, fold.asset_val, fold.scale_val)

    best_cfg = candidates[0]
    best_epoch = 1
    best_metrics = {"rmse": float("inf"), "mae": float("inf"), "sign_acc": 0.0}

    for cfg in candidates:
        set_seed(seed)
        model = RegimeAwareFuzzyHybrid(
            seq_input_dim=len(FEATURE_COLS),
            summary_input_dim=S_train.shape[1],
            n_assets=n_assets,
            channels=cfg.channels,
            summary_hidden=cfg.summary_hidden,
            expert_hidden=cfg.expert_hidden,
            n_rules=cfg.n_rules,
            dropout=cfg.dropout,
        )
        _, metrics, best_epoch_here = train_hybrid_with_early_stopping(model, train_ds, val_ds, cfg, train_cfg)
        if metrics["rmse"] < best_metrics["rmse"] - 1e-5:
            best_cfg = cfg
            best_epoch = best_epoch_here
            best_metrics = metrics
    return best_cfg, best_epoch, best_metrics


def fit_and_eval_hybrid(
    fold: FoldArrays,
    n_assets: int,
    cfg: HybridConfig,
    train_cfg: TrainConfig,
    best_epoch: int,
    seed: int,
) -> Dict[str, float]:
    X_trainval_raw = np.concatenate([fold.X_train_raw, fold.X_val_raw], axis=0)
    y_trainval_z = np.concatenate([fold.y_train_z, fold.y_val_z], axis=0)
    asset_trainval = np.concatenate([fold.asset_train, fold.asset_val], axis=0)
    scale_trainval = np.concatenate([fold.scale_train, fold.scale_val], axis=0)

    scaler = FeatureScaler().fit(X_trainval_raw)
    X_trainval = scaler.transform(X_trainval_raw)
    X_test = scaler.transform(fold.X_test_raw)

    S_trainval = summarize_windows(X_trainval)
    S_test = summarize_windows(X_test)

    train_ds = HybridDataset(X_trainval, S_trainval, y_trainval_z, asset_trainval, scale_trainval)
    test_ds = HybridDataset(X_test, S_test, fold.y_test_z, fold.asset_test, fold.scale_test)

    set_seed(seed)
    model = RegimeAwareFuzzyHybrid(
        seq_input_dim=len(FEATURE_COLS),
        summary_input_dim=S_trainval.shape[1],
        n_assets=n_assets,
        channels=cfg.channels,
        summary_hidden=cfg.summary_hidden,
        expert_hidden=cfg.expert_hidden,
        n_rules=cfg.n_rules,
        dropout=cfg.dropout,
    )
    model = fit_hybrid_fixed_epochs(model, train_ds, cfg, train_cfg, epochs=best_epoch)
    return evaluate_hybrid(model, test_ds, batch_size=train_cfg.batch_size)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
