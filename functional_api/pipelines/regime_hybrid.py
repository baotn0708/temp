from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import keras
import pandas as pd
import torch

from functional_api.core.gru import RatioSplitConfig, make_purged_ratio_split
from functional_api.core.regime_hybrid_benchmark import (
    aggregate_numeric_rows as native_aggregate_numeric_rows,
    build_window_store,
    make_purged_walk_forward_splits,
    prepare_from_indices,
    run_fold_seed_benchmark,
)
from functional_api.core.regime_hybrid import (
    configure_runtime,
    engineer_features,
    load_market_dataframe,
    run_training_for_path,
    save_artifacts,
    save_overall_summary,
    set_seed,
    summarize_result_for_overview,
    to_jsonable,
    train_one_run,
)

from functional_api.helpers import (
    build_native_report,
    ensure_output_dir,
    maybe_write_csv,
    maybe_write_json,
    maybe_write_markdown,
    price_metrics_to_rows,
    resolve_dataset_specs,
)
from functional_api.requests import RegimeHybridBenchmarkRequest, RegimeHybridTrainRequest
from functional_api.types import BenchmarkPipelineResult, TrainDatasetResult, TrainPipelineResult


@dataclass
class RegimeHybridRuntimeConfig:
    output_dir: str
    look_back: int
    train_ratio: float
    val_ratio: float
    n_mfs: int
    epochs: int
    batch_size: int
    runs: int
    seed: int
    temporal_units: int
    conv_filters: int
    dropout: float
    learning_rate: float
    component_loss_weight: float
    include_exog: bool
    max_rows: int | None
    verbose: int
    stock_name: str | None = None


@dataclass
class RegimeHybridBenchmarkRuntimeConfig:
    look_back: int
    horizon: int
    n_splits: int
    gap: int
    val_frac: float
    test_frac: float
    min_train_frac: float
    max_train_size: int | None
    n_mfs: int
    epochs: int
    batch_size: int
    temporal_units: int
    conv_filters: int
    dropout: float
    learning_rate: float
    component_loss_weight: float
    include_exog: bool
    max_rows: int | None
    verbose: int


@dataclass
class RegimeHybridFairPrepared:
    dataset_name: str
    dataset_path: Path
    split_cfg: RatioSplitConfig
    prepared: object
    config: RegimeHybridRuntimeConfig


def _force_torch_cpu_default() -> None:
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass


def _train_runtime_config(request: RegimeHybridTrainRequest, *, output_dir: str, stock_name: str | None = None):
    return RegimeHybridRuntimeConfig(
        output_dir=output_dir,
        look_back=request.look_back,
        train_ratio=request.train_ratio,
        val_ratio=request.val_ratio,
        n_mfs=request.n_mfs,
        epochs=request.epochs,
        batch_size=request.batch_size,
        runs=request.runs,
        seed=request.seed,
        temporal_units=request.temporal_units,
        conv_filters=request.conv_filters,
        dropout=request.dropout,
        learning_rate=request.learning_rate,
        component_loss_weight=request.component_loss_weight,
        include_exog=request.include_exog,
        max_rows=request.max_rows,
        verbose=request.verbose,
        stock_name=stock_name,
    )


def _fair_runtime_config(request) -> RegimeHybridRuntimeConfig:
    overrides = dict(request.model_overrides.get("regime_hybrid", {}))
    return RegimeHybridRuntimeConfig(
        output_dir="",
        look_back=request.look_back,
        train_ratio=0.70,
        val_ratio=0.20,
        n_mfs=int(overrides.get("n_mfs", 2)),
        epochs=int(overrides.get("epochs", max(request.max_epochs, 1))),
        batch_size=int(overrides.get("batch_size", max(1, min(request.batch_size, 64)))),
        runs=1,
        seed=0,
        temporal_units=int(overrides.get("temporal_units", 48)),
        conv_filters=int(overrides.get("conv_filters", 48)),
        dropout=float(overrides.get("dropout", 0.15)),
        learning_rate=float(overrides.get("learning_rate", 1e-3)),
        component_loss_weight=float(overrides.get("component_loss_weight", 0.25)),
        include_exog=request.include_exog,
        max_rows=request.max_rows,
        verbose=int(overrides.get("verbose", 0)),
        stock_name=None,
    )


class RegimeHybridPipelineAdapter:
    name = "regime_hybrid"
    benchmark_label = "regime_gated_hybrid"
    fairness_note = (
        "Regime hybrid fair benchmark uses a new purged ratio split wrapper so it can be evaluated beside the other "
        "top-level models. Its native walk-forward benchmark remains available separately."
    )

    def prepare_for_fair_benchmark(self, request):
        configure_runtime()
        _force_torch_cpu_default()
        split_cfg = RatioSplitConfig(
            split_ratio=request.split_ratio,
            seq_len=request.look_back,
            horizon=request.horizon,
            gap=request.gap,
        )
        model_cfg = _fair_runtime_config(request)
        prepared_items = []
        for spec in resolve_dataset_specs(datasets=request.datasets, files=request.files):
            df = load_market_dataframe(spec.path, max_rows=request.max_rows)
            engineered, core_feature_names, seq_feature_names = engineer_features(
                df,
                include_exog=request.include_exog,
            )
            store = build_window_store(
                engineered=engineered,
                core_feature_names=core_feature_names,
                seq_feature_names=seq_feature_names,
                look_back=request.look_back,
                stock_name=spec.name,
            )
            indices = make_purged_ratio_split(
                n_samples=len(store.X_raw),
                split_cfg=split_cfg,
            )
            prepared = prepare_from_indices(
                store=store,
                train_idx=indices.train,
                val_idx=indices.val,
                test_idx=indices.test,
                scaler_fit_idx=indices.train,
            )
            prepared_items.append(
                RegimeHybridFairPrepared(
                    dataset_name=spec.name,
                    dataset_path=spec.path,
                    split_cfg=split_cfg,
                    prepared=prepared,
                    config=model_cfg,
                )
            )
        return prepared_items

    def run_fair_train_eval(self, prepared: RegimeHybridFairPrepared, seed: int, request):
        _force_torch_cpu_default()
        keras.backend.clear_session()
        set_seed(int(seed))
        runtime_cfg = RegimeHybridRuntimeConfig(**{**prepared.config.__dict__, "seed": int(seed), "stock_name": prepared.dataset_name})
        payload = train_one_run(prepared.prepared, runtime_cfg, int(seed))
        hybrid_metrics = payload["candidate_reports"]["hybrid"]["test"]["metrics"]
        return {
            "rows": price_metrics_to_rows(
                dataset_name=prepared.dataset_name,
                seed=int(seed),
                model="regime_gated_hybrid",
                split_ratio=prepared.split_cfg.split_ratio,
                metrics=hybrid_metrics,
            ),
            "metadata": {
                "sample_counts": {
                    "train": int(len(prepared.prepared.y_train)),
                    "val": int(len(prepared.prepared.y_val)),
                    "test": int(len(prepared.prepared.y_test)),
                },
                "comparison_val": payload["comparison_val"],
                "comparison_test": payload["comparison_test"],
                "hybrid_gate_mean": payload["candidate_reports"]["hybrid"].get("gate_mean"),
            },
        }

    def run_pipeline_native_train(self, request: RegimeHybridTrainRequest):
        configure_runtime()
        _force_torch_cpu_default()
        dataset_results: list[TrainDatasetResult] = []
        output_paths: dict[str, Path] = {}
        output_root = ensure_output_dir(request.artifact_policy, request.output_dir)
        summaries = []
        with keras.device("cpu"):
            for spec in resolve_dataset_specs(datasets=request.datasets, files=request.files):
                keras.backend.clear_session()
                _force_torch_cpu_default()
                set_seed(int(request.seed))
                runtime_cfg = _train_runtime_config(request, output_dir=str(output_root), stock_name=spec.name)
                results = run_training_for_path(runtime_cfg, spec.path)
                artifact_dir = output_root / spec.name
                if request.artifact_policy.save:
                    artifact_dir = save_artifacts(runtime_cfg, results)
                summaries.append(summarize_result_for_overview(results, artifact_dir))
                dataset_results.append(
                    TrainDatasetResult(
                        dataset=spec.name,
                        payload={
                            "dataset": spec.name,
                            "path": str(spec.path),
                            "best_run_seed": results["best_run_seed"],
                            "comparison_val": results["comparison_val"],
                            "comparison_test": results["comparison_test"],
                            "run_summaries": results["run_summaries"],
                        },
                        artifact_dir=artifact_dir if request.artifact_policy.save else None,
                    )
                )
        summary_path = None
        if request.artifact_policy.save:
            summary_path = save_overall_summary(_train_runtime_config(request, output_dir=str(output_root)), summaries)
            output_paths["overall_summary"] = summary_path
        return TrainPipelineResult(
            model="regime_gated_hybrid",
            datasets=dataset_results,
            metadata={"mode": "pipeline_native_train", "output_dir": str(output_root)},
            output_paths=output_paths,
        )

    def run_pipeline_native_benchmark(self, request: RegimeHybridBenchmarkRequest):
        repo_root = Path(__file__).resolve().parents[2]
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            if request.artifact_policy.save:
                output_root = ensure_output_dir(request.artifact_policy, request.output_dir)
            else:
                temp_dir = tempfile.TemporaryDirectory(prefix="regime_native_api_")
                output_root = Path(temp_dir.name)

            cmd = [
                sys.executable,
                "-m",
                "functional_api.core.regime_hybrid_benchmark",
                "--output-dir",
                str(output_root),
                "--look-back",
                str(request.look_back),
                "--horizon",
                str(request.horizon),
                "--n-splits",
                str(request.n_splits),
                "--gap",
                str(request.gap),
                "--val-frac",
                str(request.val_frac),
                "--test-frac",
                str(request.test_frac),
                "--min-train-frac",
                str(request.min_train_frac),
                "--max-train-size",
                str(request.max_train_size),
                "--n-mfs",
                str(request.n_mfs),
                "--epochs",
                str(request.epochs),
                "--batch-size",
                str(request.batch_size),
                "--eval-seeds",
                *[str(int(seed)) for seed in request.eval_seeds],
                "--temporal-units",
                str(request.temporal_units),
                "--conv-filters",
                str(request.conv_filters),
                "--dropout",
                str(request.dropout),
                "--learning-rate",
                str(request.learning_rate),
                "--component-loss-weight",
                str(request.component_loss_weight),
                "--verbose",
                str(request.verbose),
            ]
            if request.include_exog:
                cmd.append("--include-exog")
            if request.max_rows is not None:
                cmd.extend(["--max-rows", str(request.max_rows)])
            if request.files:
                cmd.extend(["--files", *[str(item) for item in request.files]])
            elif request.datasets:
                cmd.extend(["--datasets", *[str(item) for item in request.datasets]])

            completed = subprocess.run(
                cmd,
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            rows_df = pd.read_csv(output_root / "combined_benchmark_results.csv")
            aggregate_df = pd.read_csv(output_root / "combined_benchmark_aggregate.csv")
            metadata = json.loads((output_root / "combined_benchmark_metadata.json").read_text(encoding="utf-8"))
            report = (output_root / "combined_benchmark_report.md").read_text(encoding="utf-8")
            output_paths: dict[str, Path] = {}
            if request.artifact_policy.save:
                for key, path in {
                    "rows": output_root / "combined_benchmark_results.csv",
                    "aggregate": output_root / "combined_benchmark_aggregate.csv",
                    "comparisons": output_root / "combined_benchmark_comparisons.csv",
                    "comparison_aggregate": output_root / "combined_benchmark_comparison_aggregate.csv",
                    "metadata": output_root / "combined_benchmark_metadata.json",
                    "report": output_root / "combined_benchmark_report.md",
                    "workbook": output_root / "regime_gated_anfis_price_benchmark_latest.xlsx",
                }.items():
                    if path.exists():
                        output_paths[key] = path
            else:
                metadata["subprocess_stdout"] = completed.stdout
                metadata["subprocess_stderr"] = completed.stderr

            return BenchmarkPipelineResult(
                model="regime_gated_hybrid",
                rows=rows_df,
                aggregate=aggregate_df,
                metadata=metadata,
                output_paths=output_paths,
                report_markdown=report,
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


ADAPTER = RegimeHybridPipelineAdapter()
