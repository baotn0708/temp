from __future__ import annotations

from dataclasses import dataclass, field

from functional_api.internal.lstm_only_core import (
    LSTMConfig,
    benchmark_single_dataset,
    fit_and_eval_lstm,
    fit_final_lstm,
    metrics_bundle_to_rows,
    save_training_artifacts,
    select_lstm_config,
)

from functional_api.pipelines.common_sequence import (
    build_split_cfg,
    build_train_cfg,
    prepare_sequence_fair_datasets,
    run_sequence_native_benchmark,
    run_sequence_native_train,
)
from functional_api.types import ArtifactPolicy, BenchmarkRequestBase, TrainRequestBase


@dataclass
class LSTMTrainRequest(TrainRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int = -1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    train_seed: int = 7
    fast: bool = False
    hidden_dim: int | None = None
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 1e-3
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/lstm_train")
    )


@dataclass
class LSTMBenchmarkRequest(BenchmarkRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    horizon: int = 1
    gap: int = -1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    fast: bool = False
    hidden_dim: int | None = None
    num_layers: int = 1
    dropout: float = 0.10
    lr: float = 1e-3
    artifact_policy: ArtifactPolicy = field(
        default_factory=lambda: ArtifactPolicy(output_dir="./functional_api_outputs/lstm_benchmark")
    )

    def fixed_config_kwargs(self) -> dict[str, object]:
        return {"fixed_lstm_cfg": _build_fixed_config(self)}


def _build_fixed_config(request: LSTMTrainRequest | LSTMBenchmarkRequest) -> LSTMConfig | None:
    hidden_dim = getattr(request, "hidden_dim", None)
    num_layers = getattr(request, "num_layers", 1)
    dropout = getattr(request, "dropout", 0.10)
    lr = getattr(request, "lr", 1e-3)
    if hidden_dim is None and hasattr(request, "model_overrides"):
        overrides = dict(request.model_overrides.get("lstm", {}))
        hidden_dim = overrides.get("hidden_dim")
        num_layers = int(overrides.get("num_layers", num_layers))
        dropout = float(overrides.get("dropout", dropout))
        lr = float(overrides.get("lr", lr))
    if hidden_dim is None:
        return None
    return LSTMConfig(
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
        lr=float(lr),
    )


class LSTMPipelineAdapter:
    name = "lstm"
    benchmark_label = "lstm_only"
    fairness_note = (
        "LSTM fair benchmark uses the common purged chronological split. Internal tuning budget may still differ "
        "from other models because LSTM keeps its validation-based config selection."
    )

    def prepare_for_fair_benchmark(self, request):
        split_cfg = build_split_cfg(request.split_ratio, request.seq_len, request.horizon, request.gap)
        train_cfg = build_train_cfg(request.max_epochs, request.patience, request.batch_size)
        return prepare_sequence_fair_datasets(
            datasets=request.datasets,
            files=request.files,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
            select_config_fn=select_lstm_config,
            tuning_seed=request.tuning_seed,
            fast=request.fast,
            fixed_config_kwargs={"fixed_lstm_cfg": _build_fixed_config(request)},
        )

    def run_fair_train_eval(self, prepared, seed: int, request):
        metrics = fit_and_eval_lstm(
            split=prepared.split,
            lstm_cfg=prepared.selected_cfg,
            train_cfg=prepared.train_cfg,
            seed=seed,
            epochs=prepared.selected_epoch,
        )
        rows = metrics_bundle_to_rows(
            dataset_name=prepared.dataset_name,
            seed=seed,
            split_ratio=prepared.split_cfg.split_ratio,
            metrics_bundle=metrics,
        )
        for row in rows:
            row["model"] = "lstm_only"
        return {
            "rows": rows,
            "metadata": {
                "selected_config": prepared.selected_cfg,
                "selected_epoch": prepared.selected_epoch,
                "val_metrics": prepared.val_metrics,
                "sample_counts": prepared.split.sample_counts(),
            },
        }

    def run_pipeline_native_train(self, request: LSTMTrainRequest):
        return run_sequence_native_train(
            request=request,
            model_name="lstm_only",
            default_output_dir="./functional_api_outputs/lstm_train",
            fit_final_fn=fit_final_lstm,
            save_training_artifacts_fn=save_training_artifacts,
            select_config_fn=select_lstm_config,
            fixed_config=_build_fixed_config(request),
            fixed_config_kwarg="fixed_lstm_cfg",
        )

    def run_pipeline_native_benchmark(self, request: LSTMBenchmarkRequest):
        return run_sequence_native_benchmark(
            request=request,
            model_name="lstm_only",
            default_output_dir="./functional_api_outputs/lstm_benchmark",
            benchmark_single_dataset_fn=benchmark_single_dataset,
            fixed_config_kwargs={"fixed_lstm_cfg": _build_fixed_config(request)},
        )


ADAPTER = LSTMPipelineAdapter()
