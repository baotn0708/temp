from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from functional_api.core.feature_group import MODEL_NAME, FeatureGroupAnfis703Config, flatten_price_metrics, run_feature_group_anfis_703
from functional_api.core.gru import RatioSplitConfig, make_purged_ratio_split
from functional_api.core.feature_group_model import (
    PRICE_NAMES,
    TARGET_NAMES,
    PreparedData,
    analyze_sample,
    build_model,
    compute_initial_centers,
    configure_runtime,
    engineer_features,
    evaluate_predictions,
    extract_rules,
    load_market_dataframe,
    reconstruct_ohlc,
    set_seed,
    to_jsonable,
    train_one_run,
)

from functional_api.helpers import (
    aggregate_numeric_rows,
    build_native_report,
    ensure_output_dir,
    maybe_write_csv,
    maybe_write_json,
    maybe_write_markdown,
    price_metrics_to_rows,
    resolve_dataset_specs,
)
from functional_api.requests import FeatureGroupBenchmarkRequest, FeatureGroupTrainRequest
from functional_api.types import BenchmarkPipelineResult, TrainDatasetResult, TrainPipelineResult


@dataclass
class FeatureGroupFairPrepared:
    dataset_name: str
    dataset_path: Path
    split_cfg: RatioSplitConfig
    prepared: PreparedData
    n_mfs: int
    lstm_units: int
    dropout: float
    learning_rate: float
    epochs: int
    batch_size: int
    verbose: int


def _native_cfg(request: FeatureGroupTrainRequest | FeatureGroupBenchmarkRequest, seed: int) -> FeatureGroupAnfis703Config:
    return FeatureGroupAnfis703Config(
        look_back=request.look_back,
        split_ratio=request.split_ratio,
        gap=request.gap,
        n_mfs=request.n_mfs,
        epochs=request.epochs,
        batch_size=request.batch_size,
        seed=seed,
        lstm_units=request.lstm_units,
        dropout=request.dropout,
        learning_rate=request.learning_rate,
        include_exog=request.include_exog,
        max_rows=request.max_rows,
        verbose=request.verbose,
    )


def _build_fair_prepared(
    *,
    dataset_name: str,
    dataset_path: Path,
    split_cfg: RatioSplitConfig,
    look_back: int,
    include_exog: bool,
    max_rows: int | None,
) -> PreparedData:
    df = load_market_dataframe(dataset_path, max_rows=max_rows)
    engineered, core_feature_names, seq_feature_names = engineer_features(df, include_exog=include_exog)
    seq_values = engineered[list(seq_feature_names)].to_numpy(dtype=np.float32)
    targets = engineered[TARGET_NAMES].to_numpy(dtype=np.float32)
    current_close = engineered["Close"].to_numpy(dtype=np.float32)
    current_ohlc = engineered[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float32)
    next_ohlc = engineered[["next_Open", "next_High", "next_Low", "next_Close"]].to_numpy(dtype=np.float32)
    dates = engineered["Date"].to_numpy()

    X_raw = []
    y = []
    dates_used = []
    current_close_used = []
    current_ohlc_used = []
    next_ohlc_used = []

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
    dates_arr = np.asarray(dates_used)
    current_close_arr = np.asarray(current_close_used, dtype=np.float32)
    current_ohlc_arr = np.asarray(current_ohlc_used, dtype=np.float32)
    next_ohlc_arr = np.asarray(next_ohlc_used, dtype=np.float32)

    indices = make_purged_ratio_split(
        n_samples=len(X_raw_arr),
        split_cfg=RatioSplitConfig(
            split_ratio=split_cfg.split_ratio,
            seq_len=split_cfg.seq_len,
            horizon=split_cfg.horizon,
            gap=split_cfg.gap,
        ),
    )
    train_idx, val_idx, test_idx = indices.train, indices.val, indices.test

    scaler = StandardScaler()
    scaler.fit(X_raw_arr[train_idx].reshape(-1, X_raw_arr.shape[-1]))

    def _transform(item: np.ndarray) -> np.ndarray:
        if len(item) == 0:
            return item.astype(np.float32)
        return scaler.transform(item.reshape(-1, item.shape[-1])).reshape(item.shape).astype(np.float32)

    return PreparedData(
        stock_name=dataset_name,
        dates=dates_arr,
        seq_feature_names=list(seq_feature_names),
        core_feature_names=list(core_feature_names),
        X_train=_transform(X_raw_arr[train_idx]),
        X_val=_transform(X_raw_arr[val_idx]),
        X_test=_transform(X_raw_arr[test_idx]),
        y_train=y_arr[train_idx],
        y_val=y_arr[val_idx],
        y_test=y_arr[test_idx],
        current_close_train=current_close_arr[train_idx],
        current_close_val=current_close_arr[val_idx],
        current_close_test=current_close_arr[test_idx],
        current_ohlc_test=current_ohlc_arr[test_idx],
        actual_next_ohlc_test=next_ohlc_arr[test_idx],
        feature_scaler_mean=scaler.mean_.astype(np.float32),
        feature_scaler_scale=scaler.scale_.astype(np.float32),
    )


def _feature_group_overrides(request) -> dict[str, object]:
    overrides = dict(request.model_overrides.get("feature_group", {}))
    return {
        "n_mfs": int(overrides.get("n_mfs", 2)),
        "lstm_units": int(overrides.get("lstm_units", 32)),
        "dropout": float(overrides.get("dropout", 0.2)),
        "learning_rate": float(overrides.get("learning_rate", 1e-3)),
        "epochs": int(overrides.get("epochs", max(request.max_epochs, 1))),
        "batch_size": int(overrides.get("batch_size", max(1, min(request.batch_size, 64)))),
        "verbose": int(overrides.get("verbose", 0)),
    }


class FeatureGroupPipelineAdapter:
    name = "feature_group"
    benchmark_label = MODEL_NAME
    fairness_note = (
        "Feature-group ANFIS participates in the common fair benchmark through a new purged 7/2/1 wrapper. "
        "Its pipeline-native 7/3 benchmark is still available separately."
    )

    def prepare_for_fair_benchmark(self, request):
        configure_runtime()
        split_cfg = RatioSplitConfig(
            split_ratio=request.split_ratio,
            seq_len=request.look_back,
            horizon=request.horizon,
            gap=request.gap,
        )
        model_cfg = _feature_group_overrides(request)
        prepared_items = []
        for spec in resolve_dataset_specs(datasets=request.datasets, files=request.files):
            prepared_items.append(
                FeatureGroupFairPrepared(
                    dataset_name=spec.name,
                    dataset_path=spec.path,
                    split_cfg=split_cfg,
                    prepared=_build_fair_prepared(
                        dataset_name=spec.name,
                        dataset_path=spec.path,
                        split_cfg=split_cfg,
                        look_back=request.look_back,
                        include_exog=request.include_exog,
                        max_rows=request.max_rows,
                    ),
                    **model_cfg,
                )
            )
        return prepared_items

    def run_fair_train_eval(self, prepared: FeatureGroupFairPrepared, seed: int, request):
        tf.keras.backend.clear_session()
        set_seed(seed)
        model, history, best_val_loss, train_time = train_one_run(
            prepared=prepared.prepared,
            n_mfs=prepared.n_mfs,
            lstm_units=prepared.lstm_units,
            dropout=prepared.dropout,
            learning_rate=prepared.learning_rate,
            epochs=prepared.epochs,
            batch_size=prepared.batch_size,
            run_seed=seed,
            verbose=prepared.verbose,
        )
        test_target_pred = model.predict(prepared.prepared.X_test, verbose=0)
        test_price_pred = reconstruct_ohlc(prepared.prepared.current_close_test, test_target_pred)
        test_metrics = evaluate_predictions(
            prepared.prepared.actual_next_ohlc_test,
            test_price_pred,
            prepared.prepared.current_close_test,
        )
        return {
            "rows": price_metrics_to_rows(
                dataset_name=prepared.dataset_name,
                seed=seed,
                model=MODEL_NAME,
                split_ratio=prepared.split_cfg.split_ratio,
                metrics=test_metrics,
            ),
            "metadata": {
                "best_val_loss": best_val_loss,
                "epochs_trained": len(history["loss"]),
                "train_time_sec": round(float(train_time), 2),
                "sample_counts": {
                    "train": int(len(prepared.prepared.y_train)),
                    "val": int(len(prepared.prepared.y_val)),
                    "test": int(len(prepared.prepared.y_test)),
                },
            },
        }

    def run_pipeline_native_train(self, request: FeatureGroupTrainRequest):
        configure_runtime()
        dataset_results: list[TrainDatasetResult] = []
        for spec in resolve_dataset_specs(datasets=request.datasets, files=request.files):
            tf.keras.backend.clear_session()
            set_seed(request.seed)
            cfg = _native_cfg(request, seed=request.seed)
            result = run_feature_group_anfis_703(spec.path, spec.name, cfg)
            artifact_dir = None
            if request.artifact_policy.save:
                root = ensure_output_dir(request.artifact_policy, "./functional_api_outputs/feature_group_train")
                artifact_dir = root / spec.name.lower()
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "feature_group_anfis_clean.keras"
                result["model"].save(model_path)
                maybe_write_json(
                    request.artifact_policy,
                    artifact_dir / "training_summary.json",
                    to_jsonable(
                        {
                            "dataset": spec.name,
                            "model": MODEL_NAME,
                            "split_ratio": cfg.split_ratio,
                            "gap": cfg.gap,
                            "look_back": cfg.look_back,
                            "seed": cfg.seed,
                            "test_metrics": result["test_metrics"],
                            "test_metrics_flat": flatten_price_metrics(result["test_metrics"]),
                            "epochs_trained": len(result["history"]["loss"]),
                            "train_time_sec": round(float(result["train_time"]), 2),
                        }
                    ),
                )
                maybe_write_json(request.artifact_policy, artifact_dir / "rules.json", to_jsonable(result["rules"]))
                maybe_write_json(
                    request.artifact_policy, artifact_dir / "sample_analysis.json", to_jsonable(result["sample_analysis"])
                )
                maybe_write_json(
                    request.artifact_policy, artifact_dir / "history.json", to_jsonable({"history": result["history"]})
                )
            dataset_results.append(
                TrainDatasetResult(
                    dataset=spec.name,
                    payload={
                        "dataset": spec.name,
                        "path": str(spec.path),
                        "split_ratio": cfg.split_ratio,
                        "sample_counts": {
                            "train": int(len(result["prepared"].y_train)),
                            "test": int(len(result["prepared"].y_test)),
                        },
                        "test_metrics": result["test_metrics"],
                        "test_metrics_flat": flatten_price_metrics(result["test_metrics"]),
                    },
                    artifact_dir=artifact_dir,
                )
            )
        return TrainPipelineResult(
            model=MODEL_NAME,
            datasets=dataset_results,
            metadata={"mode": "pipeline_native_train", "split_ratio": request.split_ratio, "gap": request.gap},
        )

    def run_pipeline_native_benchmark(self, request: FeatureGroupBenchmarkRequest):
        configure_runtime()
        all_rows: list[dict] = []
        dataset_metadata: dict[str, object] = {}
        for spec in resolve_dataset_specs(datasets=request.datasets, files=request.files):
            seed_runs = {}
            for seed in request.eval_seeds:
                tf.keras.backend.clear_session()
                set_seed(int(seed))
                cfg = _native_cfg(request, seed=int(seed))
                result = run_feature_group_anfis_703(spec.path, spec.name, cfg)
                rows = price_metrics_to_rows(
                    dataset_name=spec.name,
                    seed=int(seed),
                    model=MODEL_NAME,
                    split_ratio=cfg.split_ratio,
                    metrics=result["test_metrics"],
                )
                all_rows.extend(rows)
                seed_runs[str(int(seed))] = {
                    "sample_counts": {
                        "train": int(len(result["prepared"].y_train)),
                        "test": int(len(result["prepared"].y_test)),
                    },
                    "metrics": result["test_metrics"],
                    "epochs_trained": int(len(result["history"]["loss"])),
                    "train_time_sec": round(float(result["train_time"]), 2),
                }
            dataset_metadata[spec.name] = {"path": str(spec.path), "runs": seed_runs}
        rows_df = pd.DataFrame(all_rows)
        aggregate_df = aggregate_numeric_rows(rows_df, group_cols=["dataset", "model", "split_ratio", "target"])
        metadata = {
            "model": MODEL_NAME,
            "protocol": {
                "split_ratio": request.split_ratio,
                "gap": request.gap,
                "look_back": request.look_back,
                "forecast_horizon": "next trading day",
                "selection_rule": "native 7/3 fixed-config benchmark",
            },
            "datasets": dataset_metadata,
            "eval_seeds": [int(seed) for seed in request.eval_seeds],
        }
        report = build_native_report("feature_group native benchmark", rows_df, aggregate_df, metadata)
        output_paths: dict[str, Path] = {}
        if request.artifact_policy.save:
            root = ensure_output_dir(request.artifact_policy, "./functional_api_outputs/feature_group_benchmark")
            for key, path in {
                "rows": maybe_write_csv(request.artifact_policy, root / "benchmark_results.csv", rows_df),
                "aggregate": maybe_write_csv(request.artifact_policy, root / "benchmark_aggregate.csv", aggregate_df),
                "metadata": maybe_write_json(request.artifact_policy, root / "benchmark_metadata.json", to_jsonable(metadata)),
                "report": maybe_write_markdown(request.artifact_policy, root / "benchmark_report.md", report),
            }.items():
                if path is not None:
                    output_paths[key] = path
        return BenchmarkPipelineResult(
            model=MODEL_NAME,
            rows=rows_df,
            aggregate=aggregate_df,
            metadata=metadata,
            output_paths=output_paths,
            report_markdown=report,
        )


ADAPTER = FeatureGroupPipelineAdapter()
