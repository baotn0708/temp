#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid regime-switching ANFIS model.

The original "ANFIS + BiLSTM residual" architecture can let the deep branch
absorb most of the predictive power. In this design, ANFIS keeps a structural
role: it forecasts directly and also produces a regime-aware gate that decides
how much to trust ANFIS vs. a temporal expert on each sample.

This file is standalone because the current environment does not have a working
TensorFlow install. It uses Keras 3 with the torch backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pandas as pd
import torch
from keras import Model, layers, ops
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


RETURN_CLIP = 0.5
BUFFER_CLIP = 0.5

CORE_FEATURE_NAMES = [
    "close_ret",
    "open_gap",
    "high_buffer",
    "low_buffer",
    "range_pct",
    "volume_ret",
]

TARGET_NAMES = [
    "target_close_ret",
    "target_open_gap",
    "target_high_buffer",
    "target_low_buffer",
]

PRICE_NAMES = ["Open", "High", "Low", "Close"]
EXPECTED_COLAB_DATA_FILES = [
    "AMZN.csv",
    "ChinaSouth_Publishing_2010_2023.csv",
    "JPM.csv",
    "PingAn_Bank_2010_2023.csv",
    "Sinopharm_2010_2023.csv",
    "TSLA.csv",
]


@dataclass
class PreparedData:
    stock_name: str
    dates: np.ndarray
    seq_feature_names: List[str]
    core_feature_names: List[str]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    current_close_train: np.ndarray
    current_close_val: np.ndarray
    current_close_test: np.ndarray
    current_ohlc_test: np.ndarray
    actual_next_ohlc_test: np.ndarray
    feature_scaler_mean: np.ndarray
    feature_scaler_scale: np.ndarray


@dataclass
class CandidateResult:
    name: str
    train_model: Model
    infer_model: Model
    history: Dict[str, List[float]]
    best_val_monitor: float
    train_time_sec: float
    val_target_pred: np.ndarray
    val_metrics: Dict[str, object]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    keras.utils.set_random_seed(seed)


def configure_runtime() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))


def inverse_softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 1e-6)
    return np.log(np.expm1(x))


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def is_running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore

        return True
    except ImportError:
        return False


def detect_default_data_root() -> str:
    if Path("/content").exists():
        return "/content"
    if Path("/kaggle/input/datasets/onnguyntrng/dataset-ats").exists():
        return "/kaggle/input/datasets/onnguyntrng/dataset-ats"
    return "."


def detect_default_output_root() -> str:
    if Path("/content").exists():
        return "/content/outputs_regime_gated_anfis"
    return "outputs_regime_gated_anfis"


def ensure_colab_csv_uploads(data_root: str, force_upload: bool = False) -> None:
    if not is_running_in_colab():
        return

    data_dir = Path(data_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in EXPECTED_COLAB_DATA_FILES if not (data_dir / name).exists()]
    if not force_upload and not missing:
        return

    print("Colab upload mode: please upload these CSV files if they are missing:")
    for name in EXPECTED_COLAB_DATA_FILES:
        status = "FOUND" if (data_dir / name).exists() else "MISSING"
        print(f"  - {name} [{status}]")

    from google.colab import files  # type: ignore

    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No files were uploaded from Colab.")

    current_dir = Path.cwd()
    for name in uploaded.keys():
        source = current_dir / name
        target = data_dir / Path(name).name
        if source.exists() and source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)

    still_missing = [name for name in EXPECTED_COLAB_DATA_FILES if not (data_dir / name).exists()]
    if still_missing:
        raise FileNotFoundError(f"Missing required uploaded CSV files after upload: {still_missing}")


def resolve_data_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    colab_candidate = Path("/content") / path_str
    if colab_candidate.exists():
        return colab_candidate

    cwd_candidate = Path.cwd() / path_str
    if cwd_candidate.exists():
        return cwd_candidate

    nested_candidate = Path.cwd() / "ATS - nhóm 14" / path_str
    if nested_candidate.exists():
        return nested_candidate

    basename = Path(path_str).name
    recursive_matches = list(Path.cwd().rglob(basename))
    if len(recursive_matches) == 1:
        return recursive_matches[0]
    if len(recursive_matches) > 1:
        direct_csv_matches = [item for item in recursive_matches if "__MACOSX" not in str(item)]
        if len(direct_csv_matches) == 1:
            return direct_csv_matches[0]

    raise FileNotFoundError(f"Dataset not found: {path_str}")


def resolve_input_paths(path_str: str) -> List[Path]:
    raw_path = Path(path_str)
    if raw_path.suffix.lower() == ".csv":
        try:
            resolved = resolve_data_path(path_str)
        except FileNotFoundError:
            parent_str = str(raw_path.parent) if str(raw_path.parent) not in {"", "."} else path_str
            resolved = resolve_data_path(parent_str)
    else:
        resolved = resolve_data_path(path_str)

    if resolved.is_file():
        if resolved.suffix.lower() != ".csv":
            raise ValueError(f"Expected a dataset folder or CSV file, got: {resolved}")
        resolved = resolved.parent

    csv_files = sorted(
        [
            item
            for item in resolved.rglob("*.csv")
            if item.is_file() and "__MACOSX" not in str(item) and not item.name.startswith("._")
        ]
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {resolved}")
    return csv_files


def membership_labels(n_mfs: int) -> List[str]:
    if n_mfs == 2:
        return ["LOW", "HIGH"]
    if n_mfs == 3:
        return ["LOW", "MEDIUM", "HIGH"]
    return [f"MF_{idx + 1}" for idx in range(n_mfs)]


@register_keras_serializable(package="tsa")
class OrderedFeatureGroupANFIS(layers.Layer):
    def __init__(
        self,
        n_mfs: int = 2,
        output_dim: int = 4,
        name_prefix: str = "anfis",
        initial_centers: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_mfs = n_mfs
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        self.initial_centers = None if initial_centers is None else np.asarray(initial_centers, dtype=np.float32)

    def build(self, input_shape):
        n_features = int(input_shape[-1])
        self.n_features = n_features
        self.n_rules = self.n_mfs ** n_features

        if self.initial_centers is None:
            initial_centers = np.sort(
                np.random.uniform(-1.0, 1.0, size=(n_features, self.n_mfs)).astype(np.float32),
                axis=1,
            )
        else:
            initial_centers = np.sort(self.initial_centers.astype(np.float32), axis=1)

        base_init = initial_centers[:, :1]
        delta_init = np.diff(initial_centers, axis=1)
        delta_init = np.maximum(delta_init, 1e-3)
        width_init = np.full((n_features, self.n_mfs), 0.5, dtype=np.float32)

        self.center_base = self.add_weight(
            name=f"{self.name_prefix}_center_base",
            shape=(n_features, 1),
            initializer=keras.initializers.Constant(base_init),
            trainable=True,
        )
        if self.n_mfs > 1:
            self.center_delta_raw = self.add_weight(
                name=f"{self.name_prefix}_center_delta_raw",
                shape=(n_features, self.n_mfs - 1),
                initializer=keras.initializers.Constant(inverse_softplus(delta_init)),
                trainable=True,
            )
        else:
            self.center_delta_raw = None

        self.width_raw = self.add_weight(
            name=f"{self.name_prefix}_width_raw",
            shape=(n_features, self.n_mfs),
            initializer=keras.initializers.Constant(inverse_softplus(width_init)),
            trainable=True,
        )

        self.consequent_p = self.add_weight(
            name=f"{self.name_prefix}_consequent_p",
            shape=(self.n_rules, n_features, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.consequent_r = self.add_weight(
            name=f"{self.name_prefix}_consequent_r",
            shape=(self.n_rules, self.output_dim),
            initializer="zeros",
            trainable=True,
        )

        self.rule_mf_indices = ops.convert_to_tensor(
            self._compute_rule_indices(n_features, self.n_mfs),
            dtype="int32",
        )
        super().build(input_shape)

    @staticmethod
    def _compute_rule_indices(n_features: int, n_mfs: int) -> np.ndarray:
        indices = []
        for rule_idx in range(n_mfs ** n_features):
            rule_mfs = []
            temp = rule_idx
            for _ in range(n_features):
                rule_mfs.append(temp % n_mfs)
                temp //= n_mfs
            indices.append(rule_mfs)
        return np.asarray(indices, dtype=np.int32)

    def get_centers(self):
        if self.n_mfs == 1:
            return self.center_base
        gaps = ops.softplus(self.center_delta_raw) + 1e-3
        offsets = ops.cumsum(gaps, axis=1)
        return ops.concatenate([self.center_base, self.center_base + offsets], axis=1)

    def get_widths(self):
        return ops.softplus(self.width_raw) + 1e-3

    def call(self, inputs, return_details: bool = False):
        batch_size = ops.shape(inputs)[0]
        x_exp = ops.expand_dims(inputs, axis=2)
        centers = ops.expand_dims(self.get_centers(), axis=0)
        widths = ops.expand_dims(self.get_widths(), axis=0)

        memberships = ops.exp(-ops.square(x_exp - centers) / (2.0 * ops.square(widths)))

        firing = ops.ones((batch_size, self.n_rules), dtype=inputs.dtype)
        for feat_idx in range(self.n_features):
            feat_memberships = memberships[:, feat_idx, :]
            mf_indices = self.rule_mf_indices[:, feat_idx]
            rule_memberships = ops.take(feat_memberships, mf_indices, axis=1)
            firing = firing * rule_memberships

        firing_sum = ops.sum(firing, axis=1, keepdims=True) + 1e-8
        firing_norm = firing / firing_sum

        x_expanded = ops.expand_dims(ops.expand_dims(inputs, axis=1), axis=3)
        p_expanded = ops.expand_dims(self.consequent_p, axis=0)
        linear = ops.sum(x_expanded * p_expanded, axis=2)
        rule_outputs = linear + self.consequent_r

        output = ops.sum(ops.expand_dims(firing_norm, axis=2) * rule_outputs, axis=1)
        if return_details:
            return output, firing_norm, rule_outputs
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_mfs": self.n_mfs,
                "output_dim": self.output_dim,
                "name_prefix": self.name_prefix,
            }
        )
        return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANFIS-gated regime hybrid over every CSV in a dataset folder")
    parser.add_argument(
        "--data",
        type=str,
        default=detect_default_data_root(),
        help="Dataset folder containing CSV files. On Colab, uploaded CSVs in /content are picked up by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=detect_default_output_root(),
        help="Artifact directory. On Colab, defaults to /content/outputs_regime_gated_anfis",
    )
    parser.add_argument("--stock-name", type=str, default=None)
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--n-mfs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-units", type=int, default=48)
    parser.add_argument("--conv-filters", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--component-loss-weight", type=float, default=0.25)
    parser.add_argument("--include-exog", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--upload-on-colab",
        action="store_true",
        help="When running inside Colab, open the file-upload dialog before training.",
    )
    parser.add_argument("--verbose", type=int, default=0)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown notebook/runtime arguments: {unknown}")
    return args


def load_market_dataframe(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    df = df.sort_values("Date").reset_index(drop=True)
    numeric_cols = [col for col in df.columns if col != "Date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)].reset_index(drop=True)

    if max_rows is not None and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame, include_exog: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    work = df.copy()
    prev_close = work["Close"].shift(1)
    body_high = work[["Open", "Close"]].max(axis=1)
    body_low = work[["Open", "Close"]].min(axis=1)

    work["close_ret"] = (work["Close"] / prev_close - 1.0).clip(-RETURN_CLIP, RETURN_CLIP)
    work["open_gap"] = (work["Open"] / prev_close - 1.0).clip(-RETURN_CLIP, RETURN_CLIP)

    safe_high = np.maximum(work["High"].to_numpy(dtype=np.float32), body_high.to_numpy(dtype=np.float32))
    safe_low = np.minimum(work["Low"].to_numpy(dtype=np.float32), body_low.to_numpy(dtype=np.float32))
    safe_body_high = np.maximum(body_high.to_numpy(dtype=np.float32), 1e-8)
    safe_body_low = np.maximum(body_low.to_numpy(dtype=np.float32), 1e-8)

    work["high_buffer"] = np.clip(np.log(safe_high / safe_body_high), 0.0, BUFFER_CLIP)
    work["low_buffer"] = np.clip(np.log(safe_body_low / np.maximum(safe_low, 1e-8)), 0.0, BUFFER_CLIP)
    work["range_pct"] = ((work["High"] - work["Low"]) / work["Close"]).clip(0.0, RETURN_CLIP)
    work["volume_ret"] = np.log1p(work["Volume"].clip(lower=0)).diff().clip(-3.0, 3.0)

    next_close = work["Close"].shift(-1)
    next_open = work["Open"].shift(-1)
    next_high = work["High"].shift(-1)
    next_low = work["Low"].shift(-1)

    next_body_high = pd.concat([next_open, next_close], axis=1).max(axis=1)
    next_body_low = pd.concat([next_open, next_close], axis=1).min(axis=1)

    safe_next_body_high = np.maximum(next_body_high.to_numpy(dtype=np.float32), 1e-8)
    safe_next_body_low = np.maximum(next_body_low.to_numpy(dtype=np.float32), 1e-8)
    safe_next_high = np.maximum(next_high.to_numpy(dtype=np.float32), safe_next_body_high)
    safe_next_low = np.minimum(next_low.to_numpy(dtype=np.float32), safe_next_body_low)

    work["target_close_ret"] = (next_close / work["Close"] - 1.0).clip(-RETURN_CLIP, RETURN_CLIP)
    work["target_open_gap"] = (next_open / work["Close"] - 1.0).clip(-RETURN_CLIP, RETURN_CLIP)
    work["target_high_buffer"] = np.clip(np.log(safe_next_high / safe_next_body_high), 0.0, BUFFER_CLIP)
    work["target_low_buffer"] = np.clip(np.log(safe_next_body_low / np.maximum(safe_next_low, 1e-8)), 0.0, BUFFER_CLIP)

    work["next_Open"] = next_open
    work["next_High"] = next_high
    work["next_Low"] = next_low
    work["next_Close"] = next_close

    exog_cols: List[str] = []
    if include_exog:
        exog_cols = [col for col in work.columns if col.startswith("exog_")]
        if exog_cols:
            work[exog_cols] = work[exog_cols].replace([np.inf, -np.inf], np.nan)
            work[exog_cols] = work[exog_cols].ffill().bfill().fillna(0.0)

    required_cols = CORE_FEATURE_NAMES + TARGET_NAMES + ["next_Open", "next_High", "next_Low", "next_Close", "Date"]
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=required_cols).reset_index(drop=True)

    seq_feature_cols = CORE_FEATURE_NAMES + exog_cols
    return work, CORE_FEATURE_NAMES, seq_feature_cols


def build_windows(
    engineered: pd.DataFrame,
    core_feature_names: Sequence[str],
    seq_feature_names: Sequence[str],
    look_back: int,
    train_ratio: float,
    val_ratio: float,
    stock_name: str,
) -> PreparedData:
    seq_values = engineered[list(seq_feature_names)].to_numpy(dtype=np.float32)
    targets = engineered[TARGET_NAMES].to_numpy(dtype=np.float32)
    current_close = engineered["Close"].to_numpy(dtype=np.float32)
    current_ohlc = engineered[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float32)
    next_ohlc = engineered[["next_Open", "next_High", "next_Low", "next_Close"]].to_numpy(dtype=np.float32)
    dates = engineered["Date"].to_numpy()

    X_raw: List[np.ndarray] = []
    y: List[np.ndarray] = []
    dates_used: List[np.datetime64] = []
    current_close_used: List[float] = []
    current_ohlc_used: List[np.ndarray] = []
    next_ohlc_used: List[np.ndarray] = []

    for idx in range(look_back - 1, len(engineered)):
        start = idx - look_back + 1
        X_raw.append(seq_values[start : idx + 1])
        y.append(targets[idx])
        dates_used.append(dates[idx])
        current_close_used.append(current_close[idx])
        current_ohlc_used.append(current_ohlc[idx])
        next_ohlc_used.append(next_ohlc[idx])

    X_raw_arr = np.asarray(X_raw, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    current_close_arr = np.asarray(current_close_used, dtype=np.float32)
    current_ohlc_arr = np.asarray(current_ohlc_used, dtype=np.float32)
    next_ohlc_arr = np.asarray(next_ohlc_used, dtype=np.float32)
    dates_arr = np.asarray(dates_used)

    n_samples = len(X_raw_arr)
    if n_samples < 50:
        raise ValueError(f"Not enough usable windows after feature engineering: {n_samples}")

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n_samples - 1)

    if val_end <= train_end or n_samples - val_end < 1:
        raise ValueError("Invalid split sizes after windowing; adjust look_back/train_ratio/val_ratio.")

    X_train_raw = X_raw_arr[:train_end]
    X_val_raw = X_raw_arr[train_end:val_end]
    X_test_raw = X_raw_arr[val_end:]

    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))

    X_train = scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[-1])).reshape(X_train_raw.shape).astype(np.float32)
    X_val = scaler.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape).astype(np.float32)
    X_test = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape).astype(np.float32)

    return PreparedData(
        stock_name=stock_name,
        dates=dates_arr,
        seq_feature_names=list(seq_feature_names),
        core_feature_names=list(core_feature_names),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_arr[:train_end],
        y_val=y_arr[train_end:val_end],
        y_test=y_arr[val_end:],
        current_close_train=current_close_arr[:train_end],
        current_close_val=current_close_arr[train_end:val_end],
        current_close_test=current_close_arr[val_end:],
        current_ohlc_test=current_ohlc_arr[val_end:],
        actual_next_ohlc_test=next_ohlc_arr[val_end:],
        feature_scaler_mean=scaler.mean_.astype(np.float32),
        feature_scaler_scale=scaler.scale_.astype(np.float32),
    )


def compute_initial_centers(prepared: PreparedData, n_mfs: int) -> Tuple[np.ndarray, np.ndarray]:
    last_step = prepared.X_train[:, -1, :]
    core_last = last_step[:, : len(prepared.core_feature_names)]
    returns_data = core_last[:, :4]
    indicator_data = core_last[:, 4:6]

    returns_centers = np.zeros((returns_data.shape[1], n_mfs), dtype=np.float32)
    indicator_centers = np.zeros((indicator_data.shape[1], n_mfs), dtype=np.float32)

    for feat_idx in range(returns_data.shape[1]):
        km = KMeans(n_clusters=n_mfs, random_state=42, n_init=10)
        km.fit(returns_data[:, feat_idx : feat_idx + 1])
        returns_centers[feat_idx] = np.sort(km.cluster_centers_.reshape(-1))

    for feat_idx in range(indicator_data.shape[1]):
        km = KMeans(n_clusters=n_mfs, random_state=42, n_init=10)
        km.fit(indicator_data[:, feat_idx : feat_idx + 1])
        indicator_centers[feat_idx] = np.sort(km.cluster_centers_.reshape(-1))

    return returns_centers, indicator_centers


def bounded_outputs(raw_params, prefix: str):
    close_ret = layers.Lambda(lambda z: RETURN_CLIP * ops.tanh(z[:, 0:1]), name=f"{prefix}_close_ret")(raw_params)
    open_gap = layers.Lambda(lambda z: RETURN_CLIP * ops.tanh(z[:, 1:2]), name=f"{prefix}_open_gap")(raw_params)
    high_buffer = layers.Lambda(lambda z: BUFFER_CLIP * ops.sigmoid(z[:, 2:3]), name=f"{prefix}_high_buffer")(raw_params)
    low_buffer = layers.Lambda(lambda z: BUFFER_CLIP * ops.sigmoid(z[:, 3:4]), name=f"{prefix}_low_buffer")(raw_params)
    return layers.Concatenate(name=f"{prefix}_output")([close_ret, open_gap, high_buffer, low_buffer])


def anfis_backbone(
    inputs,
    n_mfs: int,
    returns_centers: np.ndarray,
    indicator_centers: np.ndarray,
    prefix: str,
):
    last_core = layers.Lambda(lambda x: x[:, -1, : len(CORE_FEATURE_NAMES)], name=f"{prefix}_last_core")(inputs)
    returns_features = layers.Lambda(lambda x: x[:, :4], name=f"{prefix}_returns_slice")(last_core)
    indicator_features = layers.Lambda(lambda x: x[:, 4:6], name=f"{prefix}_indicator_slice")(last_core)

    returns_anfis = OrderedFeatureGroupANFIS(
        n_mfs=n_mfs,
        output_dim=4,
        name_prefix=f"{prefix}_returns",
        initial_centers=returns_centers,
        name=f"{prefix}_anfis_returns",
    )(returns_features)
    indicators_anfis = OrderedFeatureGroupANFIS(
        n_mfs=n_mfs,
        output_dim=4,
        name_prefix=f"{prefix}_indicators",
        initial_centers=indicator_centers,
        name=f"{prefix}_anfis_indicators",
    )(indicator_features)

    hidden = layers.Concatenate(name=f"{prefix}_fuzzy_concat")([returns_anfis, indicators_anfis])
    hidden = layers.Dense(24, activation="tanh", name=f"{prefix}_fuzzy_hidden_1")(hidden)
    hidden = layers.Dense(16, activation="tanh", name=f"{prefix}_fuzzy_hidden_2")(hidden)
    raw = layers.Dense(4, name=f"{prefix}_raw")(hidden)
    output = bounded_outputs(raw, prefix=f"{prefix}_bounded")
    return {"hidden": hidden, "raw": raw, "output": output, "last_core": last_core}


def temporal_residual_block(x, filters: int, dilation: int, dropout: float, prefix: str):
    residual = x
    if int(x.shape[-1]) != filters:
        residual = layers.Conv1D(filters, kernel_size=1, padding="same", name=f"{prefix}_residual_proj")(residual)

    y = layers.Conv1D(filters, kernel_size=3, padding="causal", dilation_rate=dilation, name=f"{prefix}_conv1")(x)
    y = layers.LayerNormalization(name=f"{prefix}_ln1")(y)
    y = layers.Activation("swish", name=f"{prefix}_act1")(y)
    y = layers.SpatialDropout1D(dropout, name=f"{prefix}_drop1")(y)

    y = layers.Conv1D(filters, kernel_size=3, padding="causal", dilation_rate=dilation, name=f"{prefix}_conv2")(y)
    y = layers.LayerNormalization(name=f"{prefix}_ln2")(y)
    y = layers.Activation("swish", name=f"{prefix}_act2")(y)
    y = layers.SpatialDropout1D(dropout, name=f"{prefix}_drop2")(y)
    return layers.Add(name=f"{prefix}_add")([residual, y])


def temporal_backbone(inputs, conv_filters: int, temporal_units: int, dropout: float, prefix: str):
    x = temporal_residual_block(inputs, conv_filters, dilation=1, dropout=dropout, prefix=f"{prefix}_block1")
    x = temporal_residual_block(x, conv_filters, dilation=2, dropout=dropout, prefix=f"{prefix}_block2")
    x = temporal_residual_block(x, conv_filters, dilation=4, dropout=dropout, prefix=f"{prefix}_block3")
    x = layers.GRU(temporal_units, return_sequences=True, name=f"{prefix}_gru")(x)
    x = layers.LayerNormalization(name=f"{prefix}_gru_ln")(x)

    last_step = layers.Lambda(lambda t: t[:, -1, :], name=f"{prefix}_last_step")(x)
    avg_pool = layers.GlobalAveragePooling1D(name=f"{prefix}_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name=f"{prefix}_max_pool")(x)

    hidden = layers.Concatenate(name=f"{prefix}_concat")([last_step, avg_pool, max_pool])
    hidden = layers.Dense(32, activation="swish", name=f"{prefix}_hidden_1")(hidden)
    hidden = layers.Dropout(dropout, name=f"{prefix}_dropout")(hidden)
    hidden = layers.Dense(16, activation="swish", name=f"{prefix}_hidden_2")(hidden)
    raw = layers.Dense(4, name=f"{prefix}_raw")(hidden)
    output = bounded_outputs(raw, prefix=f"{prefix}_bounded")
    return {"hidden": hidden, "raw": raw, "output": output}


def build_anfis_only_model(
    look_back: int,
    n_seq_features: int,
    n_mfs: int,
    learning_rate: float,
    returns_centers: np.ndarray,
    indicator_centers: np.ndarray,
) -> Tuple[Model, Model]:
    inputs = layers.Input(shape=(look_back, n_seq_features), name="sequence_input")
    backbone = anfis_backbone(inputs, n_mfs, returns_centers, indicator_centers, prefix="anfis_only")
    train_model = Model(inputs=inputs, outputs=backbone["output"], name="anfis_only_model")
    train_model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse", metrics=["mae"])
    infer_model = train_model
    return train_model, infer_model


def build_temporal_only_model(
    look_back: int,
    n_seq_features: int,
    conv_filters: int,
    temporal_units: int,
    dropout: float,
    learning_rate: float,
) -> Tuple[Model, Model]:
    inputs = layers.Input(shape=(look_back, n_seq_features), name="sequence_input")
    backbone = temporal_backbone(inputs, conv_filters, temporal_units, dropout, prefix="temporal_only")
    train_model = Model(inputs=inputs, outputs=backbone["output"], name="temporal_only_model")
    train_model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse", metrics=["mae"])
    infer_model = train_model
    return train_model, infer_model


def build_hybrid_model(
    look_back: int,
    n_seq_features: int,
    n_mfs: int,
    conv_filters: int,
    temporal_units: int,
    dropout: float,
    learning_rate: float,
    returns_centers: np.ndarray,
    indicator_centers: np.ndarray,
    component_loss_weight: float,
) -> Tuple[Model, Model]:
    inputs = layers.Input(shape=(look_back, n_seq_features), name="sequence_input")
    anfis_branch = anfis_backbone(inputs, n_mfs, returns_centers, indicator_centers, prefix="hybrid_anfis")
    temporal_branch = temporal_backbone(inputs, conv_filters, temporal_units, dropout, prefix="hybrid_temporal")

    gate_features = layers.Concatenate(name="gate_features")([anfis_branch["hidden"], anfis_branch["last_core"]])
    gate_hidden = layers.Dense(16, activation="tanh", name="gate_hidden")(gate_features)
    gate_logits = layers.Dense(4, name="gate_logits")(gate_hidden)
    gate = layers.Lambda(lambda g: 0.1 + 0.8 * ops.sigmoid(g), name="expert_gate")(gate_logits)

    inverse_gate = layers.Lambda(lambda g: 1.0 - g, name="inverse_gate")(gate)
    gated_anfis_raw = layers.Multiply(name="gated_anfis_raw")([gate, anfis_branch["raw"]])
    gated_temporal_raw = layers.Multiply(name="gated_temporal_raw")([inverse_gate, temporal_branch["raw"]])
    hybrid_mixed_raw = layers.Add(name="hybrid_mixed_raw")([gated_anfis_raw, gated_temporal_raw])

    fusion_features = layers.Concatenate(name="fusion_features")(
        [
            gated_anfis_raw,
            gated_temporal_raw,
            gate,
            anfis_branch["output"],
            temporal_branch["output"],
            anfis_branch["last_core"],
        ]
    )
    fusion_hidden = layers.Dense(16, activation="swish", name="fusion_hidden")(fusion_features)
    fusion_residual = layers.Dense(
        4,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="fusion_residual",
    )(fusion_hidden)
    fusion_residual = layers.Lambda(lambda z: 0.25 * z, name="fusion_residual_scaled")(fusion_residual)
    hybrid_raw = layers.Add(name="hybrid_raw")([hybrid_mixed_raw, fusion_residual])
    hybrid_output = bounded_outputs(hybrid_raw, prefix="hybrid")
    anfis_component_output = layers.Lambda(lambda z: z, name="anfis_component_output")(anfis_branch["output"])
    temporal_component_output = layers.Lambda(lambda z: z, name="temporal_component_output")(temporal_branch["output"])

    train_model = Model(
        inputs=inputs,
        outputs={
            "hybrid_output": hybrid_output,
            "anfis_component_output": anfis_component_output,
            "temporal_component_output": temporal_component_output,
        },
        name="regime_gated_anfis_hybrid_train",
    )
    train_model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss={
            "hybrid_output": "mse",
            "anfis_component_output": "mse",
            "temporal_component_output": "mse",
        },
        loss_weights={
            "hybrid_output": 1.0,
            "anfis_component_output": component_loss_weight,
            "temporal_component_output": component_loss_weight,
        },
        metrics={
            "hybrid_output": ["mae"],
            "anfis_component_output": ["mae"],
            "temporal_component_output": ["mae"],
        },
    )

    infer_model = Model(
        inputs=inputs,
        outputs={
            "hybrid_output": hybrid_output,
            "anfis_component_output": anfis_component_output,
            "temporal_component_output": temporal_component_output,
            "expert_gate": gate,
            "hybrid_raw": hybrid_raw,
        },
        name="regime_gated_anfis_hybrid_infer",
    )
    return train_model, infer_model


def reconstruct_ohlc(current_close: np.ndarray, target_params: np.ndarray) -> np.ndarray:
    current_close = np.asarray(current_close, dtype=np.float32).reshape(-1, 1)
    close_ret = target_params[:, 0:1]
    open_gap = target_params[:, 1:2]
    high_buffer = target_params[:, 2:3]
    low_buffer = target_params[:, 3:4]

    pred_close = current_close * (1.0 + close_ret)
    pred_open = current_close * (1.0 + open_gap)
    body_high = np.maximum(pred_open, pred_close)
    body_low = np.minimum(pred_open, pred_close)
    pred_high = body_high * np.exp(high_buffer)
    pred_low = body_low * np.exp(-low_buffer)
    return np.concatenate([pred_open, pred_high, pred_low, pred_close], axis=1)


def evaluate_predictions(actual_ohlc: np.ndarray, pred_ohlc: np.ndarray, current_close: np.ndarray) -> Dict[str, object]:
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(PRICE_NAMES):
        actual = actual_ohlc[:, idx]
        pred = pred_ohlc[:, idx]
        mask = actual != 0
        mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0) if mask.any() else 0.0
        metrics[name] = {
            "RMSE": float(math.sqrt(mean_squared_error(actual, pred))),
            "MAE": float(mean_absolute_error(actual, pred)),
            "MAPE": mape,
            "R2": float(r2_score(actual, pred)),
        }

    close_da = float(
        np.mean(np.sign(actual_ohlc[:, 3] - current_close) == np.sign(pred_ohlc[:, 3] - current_close)) * 100.0
    )
    open_da = float(
        np.mean(np.sign(actual_ohlc[:, 0] - current_close) == np.sign(pred_ohlc[:, 0] - current_close)) * 100.0
    )
    validity_mask = (
        (pred_ohlc[:, 1] >= np.maximum(pred_ohlc[:, 0], pred_ohlc[:, 3]))
        & (pred_ohlc[:, 2] <= np.minimum(pred_ohlc[:, 0], pred_ohlc[:, 3]))
    )
    mean_price_rmse = float(np.mean([metrics[name]["RMSE"] for name in PRICE_NAMES]))

    return {
        "price_metrics": metrics,
        "close_direction_accuracy": close_da,
        "open_direction_accuracy": open_da,
        "ohlc_validity_rate": float(np.mean(validity_mask) * 100.0),
        "mean_price_rmse": mean_price_rmse,
    }


def extract_prediction(output) -> np.ndarray:
    if isinstance(output, dict):
        if "hybrid_output" in output:
            return np.asarray(output["hybrid_output"], dtype=np.float32)
        raise ValueError(f"Cannot infer prediction tensor from keys: {list(output.keys())}")
    return np.asarray(output, dtype=np.float32)


def train_single_output_candidate(
    name: str,
    train_model: Model,
    infer_model: Model,
    prepared: PreparedData,
    epochs: int,
    batch_size: int,
    verbose: int,
) -> CandidateResult:
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=18, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=7, min_lr=1e-5, verbose=0),
    ]

    start = time.time()
    history = train_model.fit(
        prepared.X_train,
        prepared.y_train,
        validation_data=(prepared.X_val, prepared.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
    train_time = time.time() - start
    best_val_monitor = float(min(history.history["val_loss"]))
    val_target_pred = extract_prediction(infer_model.predict(prepared.X_val, verbose=0))
    val_pred_ohlc = reconstruct_ohlc(prepared.current_close_val, val_target_pred)
    val_actual_ohlc = reconstruct_ohlc(prepared.current_close_val, prepared.y_val)
    val_metrics = evaluate_predictions(val_actual_ohlc, val_pred_ohlc, prepared.current_close_val)
    return CandidateResult(
        name=name,
        train_model=train_model,
        infer_model=infer_model,
        history=history.history,
        best_val_monitor=best_val_monitor,
        train_time_sec=train_time,
        val_target_pred=val_target_pred,
        val_metrics=val_metrics,
    )


def train_hybrid_candidate(
    train_model: Model,
    infer_model: Model,
    prepared: PreparedData,
    epochs: int,
    batch_size: int,
    verbose: int,
) -> CandidateResult:
    callbacks = [
        EarlyStopping(monitor="val_hybrid_output_loss", mode="min", patience=18, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_hybrid_output_loss",
            mode="min",
            factor=0.5,
            patience=7,
            min_lr=1e-5,
            verbose=0,
        ),
    ]
    train_targets = {
        "hybrid_output": prepared.y_train,
        "anfis_component_output": prepared.y_train,
        "temporal_component_output": prepared.y_train,
    }
    val_targets = {
        "hybrid_output": prepared.y_val,
        "anfis_component_output": prepared.y_val,
        "temporal_component_output": prepared.y_val,
    }

    start = time.time()
    history = train_model.fit(
        prepared.X_train,
        train_targets,
        validation_data=(prepared.X_val, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
    train_time = time.time() - start
    best_val_monitor = float(min(history.history["val_hybrid_output_loss"]))
    val_output = infer_model.predict(prepared.X_val, verbose=0)
    val_target_pred = np.asarray(val_output["hybrid_output"], dtype=np.float32)
    val_pred_ohlc = reconstruct_ohlc(prepared.current_close_val, val_target_pred)
    val_actual_ohlc = reconstruct_ohlc(prepared.current_close_val, prepared.y_val)
    val_metrics = evaluate_predictions(val_actual_ohlc, val_pred_ohlc, prepared.current_close_val)
    return CandidateResult(
        name="hybrid",
        train_model=train_model,
        infer_model=infer_model,
        history=history.history,
        best_val_monitor=best_val_monitor,
        train_time_sec=train_time,
        val_target_pred=val_target_pred,
        val_metrics=val_metrics,
    )


def extract_layer_rules(
    layer: OrderedFeatureGroupANFIS,
    feature_names: Sequence[str],
    latent_names: Sequence[str],
) -> Dict[str, object]:
    labels = membership_labels(layer.n_mfs)
    centers = ops.convert_to_numpy(layer.get_centers())
    widths = ops.convert_to_numpy(layer.get_widths())
    rule_indices = ops.convert_to_numpy(layer.rule_mf_indices)
    p = ops.convert_to_numpy(layer.consequent_p)
    r = ops.convert_to_numpy(layer.consequent_r)

    rules = []
    for rule_idx in range(layer.n_rules):
        antecedents = []
        for feat_idx, feature_name in enumerate(feature_names):
            mf_idx = int(rule_indices[rule_idx, feat_idx])
            antecedents.append(
                {
                    "feature": feature_name,
                    "label": labels[mf_idx],
                    "center": float(centers[feat_idx, mf_idx]),
                    "width": float(widths[feat_idx, mf_idx]),
                }
            )

        consequents = {}
        for out_idx, latent_name in enumerate(latent_names):
            coeffs = {feature_names[i]: float(p[rule_idx, i, out_idx]) for i in range(len(feature_names))}
            bias = float(r[rule_idx, out_idx])
            terms = [f"{bias:+.6f}"] + [f"{coeff:+.6f}*{name}" for name, coeff in coeffs.items()]
            consequents[latent_name] = {
                "bias": bias,
                "coefficients": coeffs,
                "formula": " ".join(terms).replace("+ -", "- "),
            }

        rules.append(
            {
                "rule_index": rule_idx + 1,
                "text": "IF " + " AND ".join(f"{item['feature']} is {item['label']}" for item in antecedents),
                "antecedents": antecedents,
                "consequents": consequents,
            }
        )

    return {
        "n_rules": layer.n_rules,
        "rules": rules,
        "labels_note": "Ordered centers preserve LOW/HIGH semantics.",
    }


def extract_hybrid_rules(model: Model, feature_names: Sequence[str]) -> Dict[str, object]:
    returns_layer = model.get_layer("hybrid_anfis_anfis_returns")
    indicators_layer = model.get_layer("hybrid_anfis_anfis_indicators")
    gate_dense = model.get_layer("gate_logits")
    gate_weights = gate_dense.get_weights()
    return {
        "explainability_scope": (
            "ANFIS contributes twice: direct fuzzy forecast and regime gate that weights ANFIS vs temporal expert."
        ),
        "returns_anfis": extract_layer_rules(
            returns_layer,
            feature_names[:4],
            [f"returns_latent_{idx + 1}" for idx in range(returns_layer.output_dim)],
        ),
        "indicators_anfis": extract_layer_rules(
            indicators_layer,
            feature_names[4:6],
            [f"indicators_latent_{idx + 1}" for idx in range(indicators_layer.output_dim)],
        ),
        "gate_note": "Gate values are bounded in [0.1, 0.9], so both experts always contribute.",
        "gate_dense_weights": {
            "kernel": gate_weights[0].tolist(),
            "bias": gate_weights[1].tolist(),
        },
    }


def analyze_sample(model: Model, sample: np.ndarray, feature_names: Sequence[str]) -> Dict[str, object]:
    returns_layer: OrderedFeatureGroupANFIS = model.get_layer("hybrid_anfis_anfis_returns")
    indicators_layer: OrderedFeatureGroupANFIS = model.get_layer("hybrid_anfis_anfis_indicators")
    decompose_model = Model(
        inputs=model.input,
        outputs={
            "hybrid_output": model.get_layer("hybrid_output").output,
            "anfis_component_output": model.get_layer("hybrid_anfis_bounded_output").output,
            "temporal_component_output": model.get_layer("hybrid_temporal_bounded_output").output,
            "expert_gate": model.get_layer("expert_gate").output,
            "hybrid_anfis_raw": model.get_layer("hybrid_anfis_raw").output,
            "hybrid_temporal_raw": model.get_layer("hybrid_temporal_raw").output,
            "hybrid_mixed_raw": model.get_layer("hybrid_mixed_raw").output,
            "hybrid_raw": model.get_layer("hybrid_raw").output,
        },
        name="hybrid_decompose_model",
    )

    sample = np.asarray(sample, dtype=np.float32)
    last_core = sample[:, -1, : len(CORE_FEATURE_NAMES)]
    returns_input = ops.convert_to_tensor(last_core[:, :4], dtype="float32")
    indicators_input = ops.convert_to_tensor(last_core[:, 4:6], dtype="float32")

    _, returns_firing, returns_rule_outputs = returns_layer(returns_input, return_details=True)
    _, indicators_firing, indicators_rule_outputs = indicators_layer(indicators_input, return_details=True)
    outputs = decompose_model.predict(sample, verbose=0)

    def top_rules(layer_firing, layer_rule_outputs, prefix: str, top_k: int) -> List[Dict[str, object]]:
        firing_np = ops.convert_to_numpy(layer_firing)[0]
        rule_output_np = ops.convert_to_numpy(layer_rule_outputs)[0]
        contributions = np.abs(firing_np[:, None] * rule_output_np).sum(axis=1)
        top_indices = np.argsort(contributions)[::-1][:top_k]
        return [
            {
                "rule_index": int(idx + 1),
                "firing_strength": float(firing_np[idx]),
                "contribution_score": float(contributions[idx]),
                "latent_output": rule_output_np[idx].tolist(),
                "prefix": prefix,
            }
            for idx in top_indices
        ]

    return {
        "core_feature_names": list(feature_names),
        "last_core_features_scaled": last_core[0].tolist(),
        "hybrid_prediction": outputs["hybrid_output"][0].tolist(),
        "anfis_component_prediction": outputs["anfis_component_output"][0].tolist(),
        "temporal_component_prediction": outputs["temporal_component_output"][0].tolist(),
        "expert_gate": outputs["expert_gate"][0].tolist(),
        "hybrid_anfis_raw": outputs["hybrid_anfis_raw"][0].tolist(),
        "hybrid_temporal_raw": outputs["hybrid_temporal_raw"][0].tolist(),
        "hybrid_mixed_raw": outputs["hybrid_mixed_raw"][0].tolist(),
        "hybrid_raw": outputs["hybrid_raw"][0].tolist(),
        "top_return_rules": top_rules(returns_firing, returns_rule_outputs, "returns", top_k=3),
        "top_indicator_rules": top_rules(indicators_firing, indicators_rule_outputs, "indicators", top_k=2),
    }


def prediction_report(
    infer_model: Model,
    prepared: PreparedData,
    split: str,
) -> Dict[str, object]:
    if split == "test":
        X = prepared.X_test
        current_close = prepared.current_close_test
        actual_ohlc = prepared.actual_next_ohlc_test
    elif split == "val":
        X = prepared.X_val
        current_close = prepared.current_close_val
        actual_ohlc = reconstruct_ohlc(prepared.current_close_val, prepared.y_val)
    else:
        raise ValueError(f"Unsupported split: {split}")

    raw_output = infer_model.predict(X, verbose=0)
    if isinstance(raw_output, dict):
        hybrid_output = np.asarray(raw_output.get("hybrid_output"), dtype=np.float32) if "hybrid_output" in raw_output else None
        anfis_output = np.asarray(raw_output.get("anfis_component_output"), dtype=np.float32) if "anfis_component_output" in raw_output else None
        temporal_output = np.asarray(raw_output.get("temporal_component_output"), dtype=np.float32) if "temporal_component_output" in raw_output else None
        gate_output = np.asarray(raw_output.get("expert_gate"), dtype=np.float32) if "expert_gate" in raw_output else None
        pred = hybrid_output if hybrid_output is not None else anfis_output if anfis_output is not None else temporal_output
    else:
        pred = np.asarray(raw_output, dtype=np.float32)
        anfis_output = None
        temporal_output = None
        gate_output = None

    pred_ohlc = reconstruct_ohlc(current_close, pred)
    metrics = evaluate_predictions(actual_ohlc, pred_ohlc, current_close)
    report = {
        "target_pred": pred,
        "price_pred": pred_ohlc,
        "metrics": metrics,
    }
    if anfis_output is not None:
        report["anfis_component_metrics"] = evaluate_predictions(
            actual_ohlc,
            reconstruct_ohlc(current_close, anfis_output),
            current_close,
        )
    if temporal_output is not None:
        report["temporal_component_metrics"] = evaluate_predictions(
            actual_ohlc,
            reconstruct_ohlc(current_close, temporal_output),
            current_close,
        )
    if gate_output is not None:
        report["gate_mean"] = gate_output.mean(axis=0).tolist()
        report["gate_std"] = gate_output.std(axis=0).tolist()
    return report


def comparison_summary(candidate_reports: Dict[str, Dict[str, object]], split: str) -> Dict[str, object]:
    hybrid_score = candidate_reports["hybrid"][split]["metrics"]["mean_price_rmse"]
    anfis_score = candidate_reports["anfis_only"][split]["metrics"]["mean_price_rmse"]
    temporal_score = candidate_reports["temporal_only"][split]["metrics"]["mean_price_rmse"]
    best_component = min(anfis_score, temporal_score)
    return {
        "split": split,
        "hybrid_mean_price_rmse": hybrid_score,
        "anfis_only_mean_price_rmse": anfis_score,
        "temporal_only_mean_price_rmse": temporal_score,
        "best_component_mean_price_rmse": best_component,
        "hybrid_beats_anfis_only": hybrid_score < anfis_score,
        "hybrid_beats_temporal_only": hybrid_score < temporal_score,
        "hybrid_beats_both_components": hybrid_score < best_component,
        "hybrid_improvement_vs_best_component": best_component - hybrid_score,
    }


def train_one_run(prepared: PreparedData, args: argparse.Namespace, run_seed: int) -> Dict[str, object]:
    set_seed(run_seed)
    returns_centers, indicator_centers = compute_initial_centers(prepared, n_mfs=args.n_mfs)

    anfis_train, anfis_infer = build_anfis_only_model(
        look_back=prepared.X_train.shape[1],
        n_seq_features=prepared.X_train.shape[2],
        n_mfs=args.n_mfs,
        learning_rate=args.learning_rate,
        returns_centers=returns_centers,
        indicator_centers=indicator_centers,
    )
    temporal_train, temporal_infer = build_temporal_only_model(
        look_back=prepared.X_train.shape[1],
        n_seq_features=prepared.X_train.shape[2],
        conv_filters=args.conv_filters,
        temporal_units=args.temporal_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    hybrid_train, hybrid_infer = build_hybrid_model(
        look_back=prepared.X_train.shape[1],
        n_seq_features=prepared.X_train.shape[2],
        n_mfs=args.n_mfs,
        conv_filters=args.conv_filters,
        temporal_units=args.temporal_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        returns_centers=returns_centers,
        indicator_centers=indicator_centers,
        component_loss_weight=args.component_loss_weight,
    )

    anfis_result = train_single_output_candidate(
        name="anfis_only",
        train_model=anfis_train,
        infer_model=anfis_infer,
        prepared=prepared,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    temporal_result = train_single_output_candidate(
        name="temporal_only",
        train_model=temporal_train,
        infer_model=temporal_infer,
        prepared=prepared,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    hybrid_result = train_hybrid_candidate(
        train_model=hybrid_train,
        infer_model=hybrid_infer,
        prepared=prepared,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    candidate_reports = {
        "anfis_only": {
            "val": prediction_report(anfis_result.infer_model, prepared, split="val"),
            "test": prediction_report(anfis_result.infer_model, prepared, split="test"),
        },
        "temporal_only": {
            "val": prediction_report(temporal_result.infer_model, prepared, split="val"),
            "test": prediction_report(temporal_result.infer_model, prepared, split="test"),
        },
        "hybrid": {
            "val": prediction_report(hybrid_result.infer_model, prepared, split="val"),
            "test": prediction_report(hybrid_result.infer_model, prepared, split="test"),
        },
    }
    comparison_val = comparison_summary(candidate_reports, split="val")
    comparison_test = comparison_summary(candidate_reports, split="test")

    run_score = (
        comparison_val["hybrid_improvement_vs_best_component"],
        -candidate_reports["hybrid"]["val"]["metrics"]["mean_price_rmse"],
    )

    return {
        "run_seed": run_seed,
        "run_score": run_score,
        "candidate_results": {
            "anfis_only": anfis_result,
            "temporal_only": temporal_result,
            "hybrid": hybrid_result,
        },
        "candidate_reports": candidate_reports,
        "comparison_val": comparison_val,
        "comparison_test": comparison_test,
    }


def run_training_for_path(args: argparse.Namespace, data_path: Path) -> Dict[str, object]:
    stock_name = args.stock_name or data_path.stem

    df = load_market_dataframe(data_path, max_rows=args.max_rows)
    engineered, core_feature_names, seq_feature_names = engineer_features(df, include_exog=args.include_exog)
    prepared = build_windows(
        engineered=engineered,
        core_feature_names=core_feature_names,
        seq_feature_names=seq_feature_names,
        look_back=args.look_back,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stock_name=stock_name,
    )

    best_run_payload: Optional[Dict[str, object]] = None
    run_summaries: List[Dict[str, object]] = []

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        payload = train_one_run(prepared, args, run_seed)

        summary = {
            "run": run_idx + 1,
            "seed": run_seed,
            "comparison_val": payload["comparison_val"],
            "comparison_test": payload["comparison_test"],
            "hybrid_train_time_sec": round(payload["candidate_results"]["hybrid"].train_time_sec, 2),
            "anfis_train_time_sec": round(payload["candidate_results"]["anfis_only"].train_time_sec, 2),
            "temporal_train_time_sec": round(payload["candidate_results"]["temporal_only"].train_time_sec, 2),
        }
        run_summaries.append(summary)

        if best_run_payload is None or payload["run_score"] > best_run_payload["run_score"]:
            best_run_payload = payload

    if best_run_payload is None:
        raise RuntimeError("Training did not produce a best run.")

    hybrid_infer_model = best_run_payload["candidate_results"]["hybrid"].infer_model
    rules = extract_hybrid_rules(hybrid_infer_model, core_feature_names)
    sample_analysis = analyze_sample(hybrid_infer_model, prepared.X_test[:1], core_feature_names)

    return {
        "stock_name": stock_name,
        "data_path": str(data_path),
        "prepared": prepared,
        "best_run_seed": best_run_payload["run_seed"],
        "best_models": {
            name: result.infer_model for name, result in best_run_payload["candidate_results"].items()
        },
        "best_histories": {
            name: result.history for name, result in best_run_payload["candidate_results"].items()
        },
        "candidate_reports": best_run_payload["candidate_reports"],
        "comparison_val": best_run_payload["comparison_val"],
        "comparison_test": best_run_payload["comparison_test"],
        "run_summaries": run_summaries,
        "rules": rules,
        "sample_analysis": sample_analysis,
    }


def save_artifacts(args: argparse.Namespace, results: Dict[str, object]) -> Path:
    output_root = Path(args.output_dir)
    stock_dir = output_root / results["stock_name"]
    stock_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = stock_dir / "metrics.json"
    rules_path = stock_dir / "rules.json"
    sample_path = stock_dir / "sample_analysis.json"
    config_path = stock_dir / "training_config.json"
    history_path = stock_dir / "history.json"

    hybrid_model_path = stock_dir / f"{results['stock_name']}_regime_gated_hybrid.keras"
    results["best_models"]["hybrid"].save(hybrid_model_path)

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "stock_name": results["stock_name"],
            "data_path": results["data_path"],
            "seq_feature_names": results["prepared"].seq_feature_names,
            "core_feature_names": results["prepared"].core_feature_names,
        }
    )

    metrics_payload = {
        "stock_name": results["stock_name"],
        "data_path": results["data_path"],
        "best_run_seed": results["best_run_seed"],
        "comparison_val": results["comparison_val"],
        "comparison_test": results["comparison_test"],
        "candidate_reports": {
            name: {
                split: {
                    k: v
                    for k, v in split_payload.items()
                    if k not in {"target_pred", "price_pred"}
                }
                for split, split_payload in candidate_payload.items()
            }
            for name, candidate_payload in results["candidate_reports"].items()
        },
        "run_summaries": results["run_summaries"],
        "explainability_note": results["rules"]["explainability_scope"],
    }

    history_payload = {"history": results["best_histories"]}

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics_payload), handle, indent=2)
    with open(rules_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(results["rules"]), handle, indent=2)
    with open(sample_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(results["sample_analysis"]), handle, indent=2)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(config_payload), handle, indent=2)
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(history_payload), handle, indent=2)

    return stock_dir


def summarize_result_for_overview(results: Dict[str, object], artifact_dir: Path) -> Dict[str, object]:
    hybrid_test = results["candidate_reports"]["hybrid"]["test"]["metrics"]
    comparison_test = results["comparison_test"]
    return {
        "stock_name": results["stock_name"],
        "data_path": results["data_path"],
        "artifact_dir": str(artifact_dir),
        "hybrid_mean_price_rmse": hybrid_test["mean_price_rmse"],
        "close_rmse": hybrid_test["price_metrics"]["Close"]["RMSE"],
        "close_r2": hybrid_test["price_metrics"]["Close"]["R2"],
        "close_direction_accuracy": hybrid_test["close_direction_accuracy"],
        "hybrid_beats_anfis_only": comparison_test["hybrid_beats_anfis_only"],
        "hybrid_beats_temporal_only": comparison_test["hybrid_beats_temporal_only"],
        "hybrid_beats_both_components": comparison_test["hybrid_beats_both_components"],
        "hybrid_improvement_vs_best_component": comparison_test["hybrid_improvement_vs_best_component"],
    }


def save_overall_summary(args: argparse.Namespace, summaries: List[Dict[str, object]]) -> Path:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "overall_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable({"runs": summaries}), handle, indent=2)
    return summary_path


def print_summary(results: Dict[str, object], artifact_dir: Path) -> None:
    test_metrics = results["candidate_reports"]["hybrid"]["test"]["metrics"]
    comparison_test = results["comparison_test"]
    gate_mean = results["candidate_reports"]["hybrid"]["test"].get("gate_mean", [])

    print("\n" + "=" * 88)
    print(f"REGIME-GATED ANFIS HYBRID SUMMARY - {results['stock_name']}")
    print("=" * 88)
    print(f"Best run seed: {results['best_run_seed']}")
    print(f"Artifacts: {artifact_dir}")
    print("")
    print(
        "Hybrid beats components on test: "
        f"ANFIS-only={comparison_test['hybrid_beats_anfis_only']} | "
        f"Temporal-only={comparison_test['hybrid_beats_temporal_only']}"
    )
    print(f"Hybrid improvement vs best component (mean RMSE): {comparison_test['hybrid_improvement_vs_best_component']:.6f}")
    if gate_mean:
        print(f"Mean ANFIS gate per target: {[round(x, 4) for x in gate_mean]}")
    print("")
    for name in PRICE_NAMES:
        item = test_metrics["price_metrics"][name]
        print(
            f"{name:5s} | RMSE={item['RMSE']:.4f} | MAE={item['MAE']:.4f} | "
            f"MAPE={item['MAPE']:.2f}% | R2={item['R2']:.4f}"
        )
    print("")
    print(f"Close directional accuracy: {test_metrics['close_direction_accuracy']:.2f}%")
    print(f"Open directional accuracy : {test_metrics['open_direction_accuracy']:.2f}%")
    print(f"OHLC validity rate        : {test_metrics['ohlc_validity_rate']:.2f}%")


def print_overall_summary(summaries: List[Dict[str, object]], summary_path: Path) -> None:
    print("\n" + "=" * 88)
    print("OVERALL DATASET SUMMARY")
    print("=" * 88)
    print(f"Summary file: {summary_path}")
    print("")
    for item in summaries:
        print(
            f"{item['stock_name']:20s} | "
            f"mean_RMSE={item['hybrid_mean_price_rmse']:.4f} | "
            f"Close_R2={item['close_r2']:.4f} | "
            f"beats_both={item['hybrid_beats_both_components']}"
        )


def main() -> None:
    configure_runtime()
    args = parse_args()
    set_seed(args.seed)
    ensure_colab_csv_uploads(args.data, force_upload=args.upload_on_colab)
    input_paths = resolve_input_paths(args.data)
    overall_summaries: List[Dict[str, object]] = []
    print(f"Found {len(input_paths)} CSV files under dataset folder: {args.data}")

    for data_path in input_paths:
        run_args = argparse.Namespace(**vars(args))
        if len(input_paths) > 1 or args.stock_name is None:
            run_args.stock_name = data_path.stem
        results = run_training_for_path(run_args, data_path)
        artifact_dir = save_artifacts(run_args, results)
        print_summary(results, artifact_dir)
        overall_summaries.append(summarize_result_for_overview(results, artifact_dir))

    summary_path = save_overall_summary(args, overall_summaries)
    if len(overall_summaries) > 1:
        print_overall_summary(overall_summaries, summary_path)


if __name__ == "__main__":
    main()
