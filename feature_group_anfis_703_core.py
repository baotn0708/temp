from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from run_feature_group_anfis_clean import (
    PRICE_NAMES,
    PreparedData,
    analyze_sample,
    build_model,
    compute_initial_centers,
    engineer_features,
    evaluate_predictions,
    extract_rules,
    load_market_dataframe,
    reconstruct_ohlc,
    set_seed,
)


MODEL_NAME = "feature_group_anfis_clean"


@dataclass
class FeatureGroupAnfis703Config:
    look_back: int = 60
    split_ratio: str = "7/3"
    gap: int = 1
    n_mfs: int = 2
    epochs: int = 150
    batch_size: int = 32
    seed: int = 7
    lstm_units: int = 32
    dropout: float = 0.2
    learning_rate: float = 1e-3
    include_exog: bool = False
    max_rows: int | None = None
    verbose: int = 0


def parse_ratio(split_ratio: str) -> Tuple[int, int]:
    normalized = split_ratio.replace(":", "/").replace("-", "/").replace(" ", "")
    if normalized != "7/3":
        raise ValueError("feature_group_anfis 7/3 protocol only supports split_ratio='7/3'")
    return 7, 3


def make_purged_703_split(n_samples: int, split_ratio: str, gap: int) -> Tuple[np.ndarray, np.ndarray]:
    train_part, test_part = parse_ratio(split_ratio)
    usable = n_samples - gap
    if usable < 2:
        raise ValueError(f"Not enough samples for split {split_ratio} with gap={gap}.")

    total = train_part + test_part
    train_count = max(1, int(math.floor(usable * train_part / total)))
    test_count = usable - train_count
    if test_count < 1:
        raise ValueError(f"Not enough test samples for split {split_ratio} with gap={gap}.")

    train_idx = np.arange(train_count, dtype=np.int64)
    test_start = train_count + gap
    test_idx = np.arange(test_start, test_start + test_count, dtype=np.int64)
    return train_idx, test_idx


def build_windows_703_split(
    engineered,
    core_feature_names: Sequence[str],
    seq_feature_names: Sequence[str],
    look_back: int,
    split_ratio: str,
    gap: int,
    stock_name: str,
) -> PreparedData:
    seq_values = engineered[list(seq_feature_names)].to_numpy(dtype=np.float32)
    targets = engineered[["target_close_ret", "target_open_gap", "target_high_buffer", "target_low_buffer"]].to_numpy(
        dtype=np.float32
    )
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

    train_idx, test_idx = make_purged_703_split(n_samples=n_samples, split_ratio=split_ratio, gap=gap)

    X_train_raw = X_raw_arr[train_idx]
    X_test_raw = X_raw_arr[test_idx]
    y_train = y_arr[train_idx]
    y_test = y_arr[test_idx]

    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))

    X_train = scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[-1])).reshape(X_train_raw.shape).astype(np.float32)
    X_test = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape).astype(np.float32)

    empty_X = np.empty((0, look_back, len(seq_feature_names)), dtype=np.float32)
    empty_y = np.empty((0, y_arr.shape[-1]), dtype=np.float32)
    empty_close = np.empty((0,), dtype=np.float32)

    return PreparedData(
        stock_name=stock_name,
        dates=dates_arr,
        seq_feature_names=list(seq_feature_names),
        core_feature_names=list(core_feature_names),
        X_train=X_train,
        X_val=empty_X,
        X_test=X_test,
        y_train=y_train,
        y_val=empty_y,
        y_test=y_test,
        current_close_train=current_close_arr[train_idx],
        current_close_val=empty_close,
        current_close_test=current_close_arr[test_idx],
        current_ohlc_test=current_ohlc_arr[test_idx],
        actual_next_ohlc_test=next_ohlc_arr[test_idx],
        feature_scaler_mean=scaler.mean_.astype(np.float32),
        feature_scaler_scale=scaler.scale_.astype(np.float32),
    )


def train_one_run_no_val(
    prepared: PreparedData,
    cfg: FeatureGroupAnfis703Config,
):
    set_seed(cfg.seed)
    returns_centers, indicator_centers = compute_initial_centers(prepared, n_mfs=cfg.n_mfs)
    model = build_model(
        look_back=prepared.X_train.shape[1],
        n_seq_features=prepared.X_train.shape[2],
        n_mfs=cfg.n_mfs,
        lstm_units=cfg.lstm_units,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        returns_centers=returns_centers,
        indicator_centers=indicator_centers,
    )

    start = time.time()
    history = model.fit(
        prepared.X_train,
        prepared.y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
        callbacks=[],
        shuffle=False,
    )
    train_time = time.time() - start
    return model, history.history, train_time


def flatten_price_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    price_metrics = metrics["price_metrics"]
    for price_name in PRICE_NAMES:
        lower = price_name.lower()
        item = price_metrics[price_name]
        flat[f"{lower}_rmse"] = float(item["RMSE"])
        flat[f"{lower}_mae"] = float(item["MAE"])
        flat[f"{lower}_mape"] = float(item["MAPE"])
        flat[f"{lower}_r2"] = float(item["R2"])
    flat["close_sign_acc"] = float(metrics["close_direction_accuracy"])
    flat["open_sign_acc"] = float(metrics["open_direction_accuracy"])
    flat["ohlc_validity_rate"] = float(metrics["ohlc_validity_rate"])
    return flat


def run_feature_group_anfis_703(
    data_path: str | Path,
    stock_name: str,
    cfg: FeatureGroupAnfis703Config,
) -> Dict[str, object]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = load_market_dataframe(path, max_rows=cfg.max_rows)
    engineered, core_feature_names, seq_feature_names = engineer_features(df, include_exog=cfg.include_exog)
    prepared = build_windows_703_split(
        engineered=engineered,
        core_feature_names=core_feature_names,
        seq_feature_names=seq_feature_names,
        look_back=cfg.look_back,
        split_ratio=cfg.split_ratio,
        gap=cfg.gap,
        stock_name=stock_name,
    )

    model, history, train_time = train_one_run_no_val(prepared=prepared, cfg=cfg)
    test_target_pred = model.predict(prepared.X_test, verbose=0)
    test_price_pred = reconstruct_ohlc(prepared.current_close_test, test_target_pred)
    test_metrics = evaluate_predictions(prepared.actual_next_ohlc_test, test_price_pred, prepared.current_close_test)

    sample_analysis = analyze_sample(model, prepared.X_test[:1], core_feature_names) if len(prepared.X_test) else {}
    rules = extract_rules(model, core_feature_names)

    return {
        "stock_name": stock_name,
        "prepared": prepared,
        "model": model,
        "history": history,
        "train_time": train_time,
        "test_target_pred": test_target_pred,
        "test_price_pred": test_price_pred,
        "test_metrics": test_metrics,
        "rules": rules,
        "sample_analysis": sample_analysis,
    }
