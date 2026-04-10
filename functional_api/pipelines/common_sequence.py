from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from functional_api.helpers import (
    aggregate_numeric_rows,
    build_native_report,
    ensure_output_dir,
    maybe_write_csv,
    maybe_write_json,
    maybe_write_markdown,
    resolve_dataset_specs,
)
from functional_api.types import ArtifactPolicy, BenchmarkPipelineResult, TrainDatasetResult, TrainPipelineResult
from functional_api.internal.gru_only_core import RatioSplitConfig, gather_split_arrays, load_asset_dataset
from functional_api.internal.hybrid_core import TrainConfig


@dataclass
class PreparedSequenceDataset:
    dataset_name: str
    dataset_path: Path
    split_cfg: RatioSplitConfig
    train_cfg: TrainConfig
    split: Any
    selected_cfg: Any
    selected_epoch: int
    val_metrics: dict[str, object] | None


def build_split_cfg(split_ratio: str, seq_len: int, horizon: int, gap: int) -> RatioSplitConfig:
    return RatioSplitConfig(
        split_ratio=split_ratio,
        seq_len=seq_len,
        horizon=horizon,
        gap=None if gap < 0 else gap,
    )


def build_train_cfg(max_epochs: int, patience: int, batch_size: int) -> TrainConfig:
    return TrainConfig(max_epochs=max_epochs, patience=patience, batch_size=batch_size)


def prepare_sequence_fair_datasets(
    *,
    datasets: Sequence[str] | None,
    files: Sequence[str] | None,
    split_cfg: RatioSplitConfig,
    train_cfg: TrainConfig,
    select_config_fn: Callable[..., tuple[Any, int, dict[str, object] | None]],
    tuning_seed: int,
    fast: bool,
    fixed_config_kwargs: dict[str, Any],
) -> list[PreparedSequenceDataset]:
    prepared: list[PreparedSequenceDataset] = []
    for spec in resolve_dataset_specs(datasets=datasets, files=files):
        asset = load_asset_dataset(path=spec.path, name=spec.name, split_cfg=split_cfg)
        split = gather_split_arrays(asset, split_cfg)
        selected_cfg, selected_epoch, val_metrics = select_config_fn(
            split=split,
            train_cfg=train_cfg,
            tuning_seed=tuning_seed,
            fast=fast,
            **fixed_config_kwargs,
        )
        prepared.append(
            PreparedSequenceDataset(
                dataset_name=spec.name,
                dataset_path=spec.path,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                split=split,
                selected_cfg=selected_cfg,
                selected_epoch=selected_epoch,
                val_metrics=val_metrics,
            )
        )
    return prepared


def run_sequence_native_train(
    *,
    request,
    model_name: str,
    default_output_dir: str,
    fit_final_fn: Callable[..., tuple[Any, Any, dict[str, object]]],
    save_training_artifacts_fn: Callable[..., None],
    select_config_fn: Callable[..., tuple[Any, int, dict[str, object] | None]],
    fixed_config: Any | None,
    fixed_config_kwarg: str,
) -> TrainPipelineResult:
    split_cfg = build_split_cfg(request.split_ratio, request.seq_len, request.horizon, request.gap)
    train_cfg = build_train_cfg(request.max_epochs, request.patience, request.batch_size)
    prepared = []
    for spec in resolve_dataset_specs(datasets=request.datasets, files=request.files):
        asset = load_asset_dataset(path=spec.path, name=spec.name, split_cfg=split_cfg)
        split = gather_split_arrays(asset, split_cfg)
        selected_cfg, selected_epoch, val_metrics = select_config_fn(
            split=split,
            train_cfg=train_cfg,
            tuning_seed=request.tuning_seed,
            fast=request.fast,
            **{fixed_config_kwarg: fixed_config},
        )
        model, scaler, test_metrics = fit_final_fn(
            split=split,
            train_cfg=train_cfg,
            seed=request.train_seed,
            epochs=selected_epoch,
            **{model_name.replace("_only", "") + "_cfg": selected_cfg},
        )
        artifact_dir = None
        if request.artifact_policy.save:
            root = ensure_output_dir(request.artifact_policy, default_output_dir)
            artifact_dir = root / spec.name.lower()
            save_training_artifacts_fn(
                output_dir=artifact_dir,
                dataset_name=spec.name,
                split_cfg=split_cfg,
                train_cfg=train_cfg,
                selected_epoch=selected_epoch,
                sample_counts=split.sample_counts(),
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                scaler=scaler,
                model=model,
                **{model_name.replace("_only", "") + "_cfg": selected_cfg},
            )
        prepared.append(
            TrainDatasetResult(
                dataset=spec.name,
                payload={
                    "dataset": spec.name,
                    "path": str(spec.path),
                    "selected_config": selected_cfg,
                    "selected_epoch": selected_epoch,
                    "sample_counts": split.sample_counts(),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                artifact_dir=artifact_dir,
            )
        )
    metadata = {
        "model": model_name,
        "split_ratio": split_cfg.split_ratio,
        "seq_len": split_cfg.seq_len,
        "horizon": split_cfg.horizon,
        "gap": split_cfg.effective_gap,
    }
    return TrainPipelineResult(model=model_name, datasets=prepared, metadata=metadata)


def run_sequence_native_benchmark(
    *,
    request,
    model_name: str,
    default_output_dir: str,
    benchmark_single_dataset_fn: Callable[..., dict[str, object]],
    fixed_config_kwargs: dict[str, Any] | None = None,
) -> BenchmarkPipelineResult:
    split_cfg = build_split_cfg(request.split_ratio, request.seq_len, request.horizon, request.gap)
    train_cfg = build_train_cfg(request.max_epochs, request.patience, request.batch_size)
    all_rows: list[dict] = []
    dataset_metadata: dict[str, object] = {}
    for spec in resolve_dataset_specs(datasets=request.datasets, files=request.files):
        result = benchmark_single_dataset_fn(
            dataset_name=spec.name,
            path=spec.path,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
            eval_seeds=request.eval_seeds,
            tuning_seed=request.tuning_seed,
            fast=request.fast,
            **(fixed_config_kwargs or {}),
        )
        all_rows.extend(result["rows"])
        dataset_metadata[spec.name] = {
            "path": str(spec.path),
            "selected_config": result["selected_config"],
            "selected_epoch": result["selected_epoch"],
            "sample_counts": result["sample_counts"],
            "val_metrics": result["val_metrics"],
            "has_val": result["has_val"],
        }
    import pandas as pd

    rows = pd.DataFrame(all_rows)
    aggregate = aggregate_numeric_rows(rows, group_cols=["dataset", "model", "split_ratio", "target"])
    metadata = {
        "model": model_name,
        "split_ratio": split_cfg.split_ratio,
        "seq_len": split_cfg.seq_len,
        "horizon": split_cfg.horizon,
        "gap": split_cfg.effective_gap,
        "eval_seeds": [int(seed) for seed in request.eval_seeds],
        "datasets": dataset_metadata,
    }
    report = build_native_report(f"{model_name} benchmark", rows, aggregate, metadata)
    output_paths: dict[str, Path] = {}
    if request.artifact_policy.save:
        root = ensure_output_dir(request.artifact_policy, default_output_dir)
        raw_path = maybe_write_csv(request.artifact_policy, root / "benchmark_results.csv", rows)
        agg_path = maybe_write_csv(request.artifact_policy, root / "benchmark_aggregate.csv", aggregate)
        meta_path = maybe_write_json(request.artifact_policy, root / "benchmark_metadata.json", metadata)
        report_path = maybe_write_markdown(request.artifact_policy, root / "benchmark_report.md", report)
        for key, path in {
            "rows": raw_path,
            "aggregate": agg_path,
            "metadata": meta_path,
            "report": report_path,
        }.items():
            if path is not None:
                output_paths[key] = path
    return BenchmarkPipelineResult(
        model=model_name,
        rows=rows,
        aggregate=aggregate,
        metadata=metadata,
        output_paths=output_paths,
        report_markdown=report,
    )


def save_fair_rows(
    *,
    artifact_policy: ArtifactPolicy,
    default_output_dir: str,
    model_name: str,
    rows,
    metadata: dict[str, object],
) -> dict[str, Path]:
    aggregate = aggregate_numeric_rows(rows, group_cols=["dataset", "model", "split_ratio", "target"])
    report = build_native_report(f"{model_name} fair benchmark", rows, aggregate, metadata)
    if not artifact_policy.save:
        return {}
    root = ensure_output_dir(artifact_policy, default_output_dir)
    model_root = root / model_name
    paths = {
        "rows": maybe_write_csv(artifact_policy, model_root / "fair_rows.csv", rows),
        "aggregate": maybe_write_csv(artifact_policy, model_root / "fair_aggregate.csv", aggregate),
        "metadata": maybe_write_json(artifact_policy, model_root / "fair_metadata.json", metadata),
        "report": maybe_write_markdown(artifact_policy, model_root / "fair_report.md", report),
    }
    return {key: path for key, path in paths.items() if path is not None}
