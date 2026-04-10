from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .gru import resolve_datasets
from .regime_hybrid import (
    PRICE_NAMES,
    PreparedData,
    build_anfis_only_model,
    build_hybrid_model,
    build_temporal_only_model,
    comparison_summary,
    compute_initial_centers,
    configure_runtime,
    engineer_features,
    load_market_dataframe,
    prediction_report,
    set_seed,
    to_jsonable,
    train_hybrid_candidate,
    train_single_output_candidate,
)


MODEL_NAMES = ["anfis_only", "temporal_only", "hybrid"]


@dataclass
class FoldIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class WindowStore:
    stock_name: str
    dates: np.ndarray
    seq_feature_names: List[str]
    core_feature_names: List[str]
    X_raw: np.ndarray
    y: np.ndarray
    current_close: np.ndarray
    current_ohlc: np.ndarray
    next_ohlc: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean walk-forward benchmark for the regime-gated ANFIS hybrid with original OHLC price reconstruction."
    )
    parser.add_argument("--datasets", nargs="+", default=["AMZN", "JPM", "TSLA"])
    parser.add_argument("--files", nargs="*", default=None, help="Optional CSV paths. If set, --datasets is ignored.")
    parser.add_argument("--output-dir", type=str, default="./regime_gated_anfis_price_benchmark_outputs")
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--gap", type=int, default=-1, help="Use -1 to default gap=horizon.")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--min-train-frac", type=float, default=0.40)
    parser.add_argument("--max-train-size", type=int, default=768)
    parser.add_argument("--n-mfs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--temporal-units", type=int, default=48)
    parser.add_argument("--conv-filters", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--component-loss-weight", type=float, default=0.25)
    parser.add_argument("--include-exog", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def build_window_store(
    engineered: pd.DataFrame,
    core_feature_names: Sequence[str],
    seq_feature_names: Sequence[str],
    look_back: int,
    stock_name: str,
) -> WindowStore:
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

    if len(X_raw_arr) < 50:
        raise ValueError(f"Not enough usable windows after feature engineering: {len(X_raw_arr)}")

    return WindowStore(
        stock_name=stock_name,
        dates=dates_arr,
        seq_feature_names=list(seq_feature_names),
        core_feature_names=list(core_feature_names),
        X_raw=X_raw_arr,
        y=y_arr,
        current_close=current_close_arr,
        current_ohlc=current_ohlc_arr,
        next_ohlc=next_ohlc_arr,
    )


def make_purged_walk_forward_splits(
    n_samples: int,
    n_splits: int,
    val_frac: float,
    test_frac: float,
    gap: int,
    min_train_frac: float,
    max_train_size: Optional[int],
) -> List[FoldIndices]:
    test_size = max(1, int(round(n_samples * test_frac)))
    val_size = max(1, int(round(n_samples * val_frac)))
    first_test_start = max(int(round(n_samples * min_train_frac)), n_samples - n_splits * test_size)

    folds: List[FoldIndices] = []
    for fold_idx in range(n_splits):
        test_start = first_test_start + fold_idx * test_size
        test_end = min(test_start + test_size, n_samples)

        val_end = test_start - gap
        val_start = max(0, val_end - val_size)

        train_end = val_start - gap
        train_start = 0 if max_train_size is None else max(0, train_end - max_train_size)

        if train_end <= train_start or val_start >= val_end or test_start >= test_end:
            continue

        folds.append(
            FoldIndices(
                train=np.arange(train_start, train_end, dtype=np.int64),
                val=np.arange(val_start, val_end, dtype=np.int64),
                test=np.arange(test_start, test_end, dtype=np.int64),
            )
        )
    return folds


def _transform_windows(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X.astype(np.float32)
    reshaped = X.reshape(-1, X.shape[-1])
    return scaler.transform(reshaped).reshape(X.shape).astype(np.float32)


def prepare_from_indices(
    store: WindowStore,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler_fit_idx: np.ndarray,
) -> PreparedData:
    X_fit = store.X_raw[scaler_fit_idx]
    scaler = StandardScaler()
    scaler.fit(X_fit.reshape(-1, X_fit.shape[-1]))

    X_train = _transform_windows(scaler, store.X_raw[train_idx])
    X_val = _transform_windows(scaler, store.X_raw[val_idx])
    X_test = _transform_windows(scaler, store.X_raw[test_idx])

    return PreparedData(
        stock_name=store.stock_name,
        dates=store.dates,
        seq_feature_names=list(store.seq_feature_names),
        core_feature_names=list(store.core_feature_names),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=store.y[train_idx],
        y_val=store.y[val_idx],
        y_test=store.y[test_idx],
        current_close_train=store.current_close[train_idx],
        current_close_val=store.current_close[val_idx],
        current_close_test=store.current_close[test_idx],
        current_ohlc_test=store.current_ohlc[test_idx],
        actual_next_ohlc_test=store.next_ohlc[test_idx],
        feature_scaler_mean=scaler.mean_.astype(np.float32),
        feature_scaler_scale=scaler.scale_.astype(np.float32),
    )


def _best_epoch(history: Dict[str, List[float]], monitor_key: str) -> int:
    values = history.get(monitor_key, [])
    if not values:
        return max(1, len(history.get("loss", [])))
    return int(np.argmin(np.asarray(values, dtype=np.float32)) + 1)


def fit_single_output_fixed_epochs(
    train_model,
    prepared: PreparedData,
    epochs: int,
    batch_size: int,
    verbose: int,
) -> Tuple[Dict[str, List[float]], float]:
    start = time.time()
    history = train_model.fit(
        prepared.X_train,
        prepared.y_train,
        epochs=max(1, int(epochs)),
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[],
        shuffle=True,
    )
    return history.history, time.time() - start


def fit_hybrid_fixed_epochs(
    train_model,
    prepared: PreparedData,
    epochs: int,
    batch_size: int,
    verbose: int,
) -> Tuple[Dict[str, List[float]], float]:
    train_targets = {
        "hybrid_output": prepared.y_train,
        "anfis_component_output": prepared.y_train,
        "temporal_component_output": prepared.y_train,
    }
    start = time.time()
    history = train_model.fit(
        prepared.X_train,
        train_targets,
        epochs=max(1, int(epochs)),
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[],
        shuffle=True,
    )
    return history.history, time.time() - start


def flatten_price_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for price_name in PRICE_NAMES:
        item = metrics["price_metrics"][price_name]
        lower = price_name.lower()
        flat[f"{lower}_rmse"] = float(item["RMSE"])
        flat[f"{lower}_mae"] = float(item["MAE"])
        flat[f"{lower}_mape"] = float(item["MAPE"])
        flat[f"{lower}_r2"] = float(item["R2"])
    flat["mean_price_rmse"] = float(metrics["mean_price_rmse"])
    flat["close_direction_accuracy"] = float(metrics["close_direction_accuracy"])
    flat["open_direction_accuracy"] = float(metrics["open_direction_accuracy"])
    flat["ohlc_validity_rate"] = float(metrics["ohlc_validity_rate"])
    return flat


def aggregate_numeric_rows(rows_df: pd.DataFrame, group_cols: Sequence[str], skip_cols: Sequence[str]) -> pd.DataFrame:
    excluded = set(group_cols) | set(skip_cols)
    metric_cols = [col for col in rows_df.columns if col not in excluded and pd.api.types.is_numeric_dtype(rows_df[col])]
    grouped = rows_df.groupby(list(group_cols), as_index=False)[metric_cols].agg(["mean", "std"])
    renamed = []
    for col in grouped.columns.to_flat_index():
        if isinstance(col, str):
            renamed.append(col)
        elif col[1] == "":
            renamed.append(col[0])
        else:
            renamed.append(f"{col[0]}_{'avg_over_runs' if col[1] == 'mean' else 'std_over_runs'}")
    grouped.columns = renamed
    return grouped


def _safe_sheet_name(name: str) -> str:
    invalid = '[]:*?/\\'
    cleaned = "".join("_" if ch in invalid else ch for ch in name)
    return cleaned[:31]


def export_workbook(
    output_root: Path,
    all_results: pd.DataFrame,
    all_aggregate: pd.DataFrame,
    all_comparisons: pd.DataFrame,
    all_comparison_aggregate: pd.DataFrame,
    sources_df: pd.DataFrame,
) -> Path:
    workbook_path = output_root / "regime_gated_anfis_price_benchmark_latest.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        all_results.to_excel(writer, sheet_name="all_results", index=False)
        all_aggregate.to_excel(writer, sheet_name="all_aggregate", index=False)
        all_comparisons.to_excel(writer, sheet_name="all_comparisons", index=False)
        all_comparison_aggregate.to_excel(writer, sheet_name="all_comp_aggregate", index=False)
        sources_df.to_excel(writer, sheet_name="sources", index=False)

        for dataset_name, dataset_df in all_results.groupby("dataset", sort=False):
            slug = dataset_name.lower()
            dataset_df.to_excel(writer, sheet_name=_safe_sheet_name(f"{slug}_results"), index=False)
            dataset_agg = all_aggregate[all_aggregate["dataset"] == dataset_name].reset_index(drop=True)
            dataset_cmp = all_comparisons[all_comparisons["dataset"] == dataset_name].reset_index(drop=True)
            dataset_cmp_agg = all_comparison_aggregate[
                all_comparison_aggregate["dataset"] == dataset_name
            ].reset_index(drop=True)
            dataset_agg.to_excel(writer, sheet_name=_safe_sheet_name(f"{slug}_aggregate"), index=False)
            dataset_cmp.to_excel(writer, sheet_name=_safe_sheet_name(f"{slug}_comparisons"), index=False)
            dataset_cmp_agg.to_excel(writer, sheet_name=_safe_sheet_name(f"{slug}_comp_agg"), index=False)
    return workbook_path


def frame_to_report_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    return df.to_string(index=False)


def build_models_for_prepared(prepared: PreparedData, args: argparse.Namespace, seed: int):
    returns_centers, indicator_centers = compute_initial_centers(prepared, n_mfs=args.n_mfs)

    set_seed(seed)
    anfis_train, anfis_infer = build_anfis_only_model(
        look_back=prepared.X_train.shape[1],
        n_seq_features=prepared.X_train.shape[2],
        n_mfs=args.n_mfs,
        learning_rate=args.learning_rate,
        returns_centers=returns_centers,
        indicator_centers=indicator_centers,
    )

    set_seed(seed)
    temporal_train, temporal_infer = build_temporal_only_model(
        look_back=prepared.X_train.shape[1],
        n_seq_features=prepared.X_train.shape[2],
        conv_filters=args.conv_filters,
        temporal_units=args.temporal_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )

    set_seed(seed)
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

    return {
        "anfis_only": (anfis_train, anfis_infer),
        "temporal_only": (temporal_train, temporal_infer),
        "hybrid": (hybrid_train, hybrid_infer),
    }


def run_fold_seed_benchmark(
    store: WindowStore,
    fold_idx: int,
    split: FoldIndices,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[List[dict], dict]:
    prepared_select = prepare_from_indices(
        store=store,
        train_idx=split.train,
        val_idx=split.val,
        test_idx=split.test,
        scaler_fit_idx=split.train,
    )
    selection_models = build_models_for_prepared(prepared_select, args, seed)

    anfis_selected = train_single_output_candidate(
        name="anfis_only",
        train_model=selection_models["anfis_only"][0],
        infer_model=selection_models["anfis_only"][1],
        prepared=prepared_select,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    temporal_selected = train_single_output_candidate(
        name="temporal_only",
        train_model=selection_models["temporal_only"][0],
        infer_model=selection_models["temporal_only"][1],
        prepared=prepared_select,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    hybrid_selected = train_hybrid_candidate(
        train_model=selection_models["hybrid"][0],
        infer_model=selection_models["hybrid"][1],
        prepared=prepared_select,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    best_epochs = {
        "anfis_only": _best_epoch(anfis_selected.history, "val_loss"),
        "temporal_only": _best_epoch(temporal_selected.history, "val_loss"),
        "hybrid": _best_epoch(hybrid_selected.history, "val_hybrid_output_loss"),
    }
    best_val_monitors = {
        "anfis_only": float(anfis_selected.best_val_monitor),
        "temporal_only": float(temporal_selected.best_val_monitor),
        "hybrid": float(hybrid_selected.best_val_monitor),
    }

    trainval_idx = np.concatenate([split.train, split.val])
    prepared_refit = prepare_from_indices(
        store=store,
        train_idx=trainval_idx,
        val_idx=np.asarray([], dtype=np.int64),
        test_idx=split.test,
        scaler_fit_idx=trainval_idx,
    )
    refit_models = build_models_for_prepared(prepared_refit, args, seed)

    anfis_history, anfis_train_time = fit_single_output_fixed_epochs(
        refit_models["anfis_only"][0],
        prepared_refit,
        epochs=best_epochs["anfis_only"],
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    temporal_history, temporal_train_time = fit_single_output_fixed_epochs(
        refit_models["temporal_only"][0],
        prepared_refit,
        epochs=best_epochs["temporal_only"],
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    hybrid_history, hybrid_train_time = fit_hybrid_fixed_epochs(
        refit_models["hybrid"][0],
        prepared_refit,
        epochs=best_epochs["hybrid"],
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    candidate_reports = {
        "anfis_only": {"test": prediction_report(refit_models["anfis_only"][1], prepared_refit, split="test")},
        "temporal_only": {"test": prediction_report(refit_models["temporal_only"][1], prepared_refit, split="test")},
        "hybrid": {"test": prediction_report(refit_models["hybrid"][1], prepared_refit, split="test")},
    }
    comparison_test = comparison_summary(candidate_reports, split="test")

    common = {
        "dataset": store.stock_name,
        "fold": int(fold_idx),
        "seed": int(seed),
        "protocol": "purged_walk_forward_price",
        "look_back": int(args.look_back),
        "horizon": int(args.horizon),
        "gap": int(args.gap if args.gap >= 0 else args.horizon),
        "train_samples": int(len(split.train)),
        "val_samples": int(len(split.val)),
        "trainval_samples": int(len(trainval_idx)),
        "test_samples": int(len(split.test)),
    }

    rows = []
    model_payloads = {
        "anfis_only": (
            candidate_reports["anfis_only"]["test"]["metrics"],
            best_epochs["anfis_only"],
            best_val_monitors["anfis_only"],
            len(anfis_history.get("loss", [])),
            anfis_train_time,
        ),
        "temporal_only": (
            candidate_reports["temporal_only"]["test"]["metrics"],
            best_epochs["temporal_only"],
            best_val_monitors["temporal_only"],
            len(temporal_history.get("loss", [])),
            temporal_train_time,
        ),
        "hybrid": (
            candidate_reports["hybrid"]["test"]["metrics"],
            best_epochs["hybrid"],
            best_val_monitors["hybrid"],
            len(hybrid_history.get("loss", [])),
            hybrid_train_time,
        ),
    }
    for model_name, (metrics, best_epoch, best_val_monitor, epochs_trained, train_time_sec) in model_payloads.items():
        row = {
            **common,
            "model": model_name,
            "best_epoch_from_val": int(best_epoch),
            "best_val_monitor": float(best_val_monitor),
            "refit_epochs": int(epochs_trained),
            "train_time_sec": round(float(train_time_sec), 2),
            **flatten_price_metrics(metrics),
        }
        if model_name == "hybrid":
            row["gate_mean_open"] = float(candidate_reports["hybrid"]["test"].get("gate_mean", [np.nan] * 4)[0])
            row["gate_mean_high"] = float(candidate_reports["hybrid"]["test"].get("gate_mean", [np.nan] * 4)[1])
            row["gate_mean_low"] = float(candidate_reports["hybrid"]["test"].get("gate_mean", [np.nan] * 4)[2])
            row["gate_mean_close"] = float(candidate_reports["hybrid"]["test"].get("gate_mean", [np.nan] * 4)[3])
        rows.append(row)

    comparison_row = {
        **common,
        **comparison_test,
    }
    return rows, comparison_row


def build_sources_df(output_root: Path) -> pd.DataFrame:
    source_paths = [
        ("all_results", output_root / "combined_benchmark_results.csv"),
        ("all_aggregate", output_root / "combined_benchmark_aggregate.csv"),
        ("all_comparisons", output_root / "combined_benchmark_comparisons.csv"),
        ("all_comp_aggregate", output_root / "combined_benchmark_comparison_aggregate.csv"),
        ("report", output_root / "combined_benchmark_report.md"),
        ("metadata", output_root / "combined_benchmark_metadata.json"),
    ]
    rows = []
    for scope, path in source_paths:
        if path.exists():
            rows.append(
                {
                    "scope": scope,
                    "path": str(path.resolve()),
                    "mtime": pd.Timestamp(path.stat().st_mtime, unit="s"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    configure_runtime()
    args = parse_args()
    datasets = resolve_datasets(args.datasets, args.files)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    effective_gap = args.horizon if args.gap < 0 else args.gap

    all_rows: List[dict] = []
    all_comparisons: List[dict] = []
    dataset_metadata: Dict[str, object] = {
        "protocol": {
            "type": "purged_walk_forward_price",
            "look_back": args.look_back,
            "horizon": args.horizon,
            "gap": effective_gap,
            "n_splits": args.n_splits,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "min_train_frac": args.min_train_frac,
            "max_train_size": args.max_train_size,
            "selection_rule": "best epoch chosen on validation only; final model refit on train+val; test evaluated once",
            "eval_seeds": [int(seed) for seed in args.eval_seeds],
        },
        "model_config": {
            "n_mfs": args.n_mfs,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "temporal_units": args.temporal_units,
            "conv_filters": args.conv_filters,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "component_loss_weight": args.component_loss_weight,
            "include_exog": args.include_exog,
            "max_rows": args.max_rows,
        },
        "datasets": {},
    }

    with keras.device("cpu"):
        for dataset_name, path in datasets:
            dataset_output = output_root / dataset_name.lower()
            dataset_output.mkdir(parents=True, exist_ok=True)

            df = load_market_dataframe(Path(path), max_rows=args.max_rows)
            engineered, core_feature_names, seq_feature_names = engineer_features(df, include_exog=args.include_exog)
            store = build_window_store(
                engineered=engineered,
                core_feature_names=core_feature_names,
                seq_feature_names=seq_feature_names,
                look_back=args.look_back,
                stock_name=dataset_name,
            )
            folds = make_purged_walk_forward_splits(
                n_samples=len(store.X_raw),
                n_splits=args.n_splits,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                gap=effective_gap,
                min_train_frac=args.min_train_frac,
                max_train_size=args.max_train_size,
            )
            if not folds:
                raise ValueError(f"No valid walk-forward splits for dataset {dataset_name}.")

            dataset_rows: List[dict] = []
            dataset_comparisons: List[dict] = []
            dataset_metadata["datasets"][dataset_name] = {
                "path": str(Path(path).resolve()),
                "n_rows_raw": int(len(df)),
                "n_windows": int(len(store.X_raw)),
                "folds": [],
            }

            for fold_idx, split in enumerate(folds):
                dataset_metadata["datasets"][dataset_name]["folds"].append(
                    {
                        "fold": int(fold_idx),
                        "train_samples": int(len(split.train)),
                        "val_samples": int(len(split.val)),
                        "test_samples": int(len(split.test)),
                        "train_start_date": str(store.dates[split.train[0]]),
                        "train_end_date": str(store.dates[split.train[-1]]),
                        "val_start_date": str(store.dates[split.val[0]]),
                        "val_end_date": str(store.dates[split.val[-1]]),
                        "test_start_date": str(store.dates[split.test[0]]),
                        "test_end_date": str(store.dates[split.test[-1]]),
                    }
                )
                for seed in args.eval_seeds:
                    rows, comparison_row = run_fold_seed_benchmark(store, fold_idx, split, args, int(seed))
                    dataset_rows.extend(rows)
                    dataset_comparisons.append(comparison_row)
                    print(
                        f"[{dataset_name}] fold={fold_idx} seed={seed} "
                        f"hybrid_close_rmse={rows[-1]['close_rmse']:.6f} "
                        f"hybrid_validity={rows[-1]['ohlc_validity_rate']:.2f}"
                    )

            dataset_rows_df = pd.DataFrame(dataset_rows)
            dataset_comparisons_df = pd.DataFrame(dataset_comparisons)
            dataset_aggregate_df = aggregate_numeric_rows(
                dataset_rows_df,
                group_cols=["dataset", "model", "protocol"],
                skip_cols=[],
            )
            dataset_comparison_aggregate_df = aggregate_numeric_rows(
                dataset_comparisons_df,
                group_cols=["dataset", "split"],
                skip_cols=[],
            )

            dataset_rows_df.to_csv(dataset_output / "benchmark_results.csv", index=False)
            dataset_aggregate_df.to_csv(dataset_output / "benchmark_aggregate.csv", index=False)
            dataset_comparisons_df.to_csv(dataset_output / "benchmark_comparisons.csv", index=False)
            dataset_comparison_aggregate_df.to_csv(dataset_output / "benchmark_comparison_aggregate.csv", index=False)
            (dataset_output / "benchmark_metadata.json").write_text(
                json.dumps(to_jsonable(dataset_metadata["datasets"][dataset_name]), indent=2),
                encoding="utf-8",
            )

            report_lines = [
                f"# Regime-gated ANFIS price benchmark: {dataset_name}",
                "",
                "## Protocol",
                f"- look_back = {args.look_back}",
                f"- horizon = {args.horizon}",
                f"- gap = {effective_gap}",
                f"- n_splits = {args.n_splits}",
                f"- val_frac = {args.val_frac}",
                f"- test_frac = {args.test_frac}",
                f"- max_train_size = {args.max_train_size}",
                f"- eval_seeds = {args.eval_seeds}",
                "- target space = original OHLC parameterization from run_regime_gated_anfis_hybrid.py",
                "- selection = best epoch on validation only, then refit on train+val",
                "",
                "## Raw benchmark rows",
                "",
                frame_to_report_table(dataset_rows_df),
                "",
                "## Aggregate over folds x seeds",
                "",
                frame_to_report_table(dataset_aggregate_df),
                "",
                "## Hybrid vs components",
                "",
                frame_to_report_table(dataset_comparisons_df),
            ]
            (dataset_output / "benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")

            all_rows.extend(dataset_rows)
            all_comparisons.extend(dataset_comparisons)

    all_results_df = pd.DataFrame(all_rows)
    all_comparisons_df = pd.DataFrame(all_comparisons)
    all_aggregate_df = aggregate_numeric_rows(
        all_results_df,
        group_cols=["dataset", "model", "protocol"],
        skip_cols=[],
    )
    all_comparison_aggregate_df = aggregate_numeric_rows(
        all_comparisons_df,
        group_cols=["dataset", "split"],
        skip_cols=[],
    )

    all_results_df.to_csv(output_root / "combined_benchmark_results.csv", index=False)
    all_aggregate_df.to_csv(output_root / "combined_benchmark_aggregate.csv", index=False)
    all_comparisons_df.to_csv(output_root / "combined_benchmark_comparisons.csv", index=False)
    all_comparison_aggregate_df.to_csv(output_root / "combined_benchmark_comparison_aggregate.csv", index=False)
    (output_root / "combined_benchmark_metadata.json").write_text(
        json.dumps(to_jsonable(dataset_metadata), indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Combined regime-gated ANFIS price benchmark",
        "",
        "## Aggregate over folds x seeds",
        "",
        frame_to_report_table(all_aggregate_df),
        "",
        "## Hybrid vs components aggregate",
        "",
        frame_to_report_table(all_comparison_aggregate_df),
    ]
    (output_root / "combined_benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    sources_df = build_sources_df(output_root)
    workbook_path = export_workbook(
        output_root=output_root,
        all_results=all_results_df,
        all_aggregate=all_aggregate_df,
        all_comparisons=all_comparisons_df,
        all_comparison_aggregate=all_comparison_aggregate_df,
        sources_df=sources_df,
    )

    print("\nCombined aggregate:")
    print(all_aggregate_df.to_string(index=False))
    print(f"\nExcel workbook: {workbook_path}")


if __name__ == "__main__":
    main()
