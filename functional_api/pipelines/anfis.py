from __future__ import annotations

from functional_api.core.anfis import (
    ANFISConfig,
    benchmark_single_dataset,
    fit_and_eval_anfis,
    fit_final_anfis,
    metrics_bundle_to_rows,
    save_training_artifacts,
    select_anfis_config,
)

from functional_api.pipelines.common_sequence import (
    build_split_cfg,
    build_train_cfg,
    prepare_sequence_fair_datasets,
    run_sequence_native_benchmark,
    run_sequence_native_train,
)
from functional_api.requests import ANFISBenchmarkRequest, ANFISTrainRequest


def _build_fixed_config(request: ANFISTrainRequest | ANFISBenchmarkRequest) -> ANFISConfig | None:
    n_rules = getattr(request, "n_rules", None)
    lr = getattr(request, "lr", 1e-3)
    if n_rules is None and hasattr(request, "model_overrides"):
        overrides = dict(request.model_overrides.get("anfis", {}))
        n_rules = overrides.get("n_rules")
        lr = float(overrides.get("lr", lr))
    if n_rules is None:
        return None
    return ANFISConfig(n_rules=int(n_rules), lr=float(lr))


class ANFISPipelineAdapter:
    name = "anfis"
    benchmark_label = "anfis_only"
    fairness_note = (
        "ANFIS fair benchmark uses the common purged chronological split. Internal tuning budget may still differ "
        "from other models because ANFIS keeps its validation-based config selection."
    )

    def prepare_for_fair_benchmark(self, request):
        split_cfg = build_split_cfg(request.split_ratio, request.seq_len, request.horizon, request.gap)
        train_cfg = build_train_cfg(request.max_epochs, request.patience, request.batch_size)
        return prepare_sequence_fair_datasets(
            datasets=request.datasets,
            files=request.files,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
            select_config_fn=select_anfis_config,
            tuning_seed=request.tuning_seed,
            fast=request.fast,
            fixed_config_kwargs={"fixed_anfis_cfg": _build_fixed_config(request)},
        )

    def run_fair_train_eval(self, prepared, seed: int, request):
        metrics = fit_and_eval_anfis(
            split=prepared.split,
            anfis_cfg=prepared.selected_cfg,
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
        return {
            "rows": rows,
            "metadata": {
                "selected_config": prepared.selected_cfg,
                "selected_epoch": prepared.selected_epoch,
                "val_metrics": prepared.val_metrics,
                "sample_counts": prepared.split.sample_counts(),
            },
        }

    def run_pipeline_native_train(self, request: ANFISTrainRequest):
        return run_sequence_native_train(
            request=request,
            model_name="anfis_only",
            default_output_dir="./functional_api_outputs/anfis_train",
            fit_final_fn=fit_final_anfis,
            save_training_artifacts_fn=save_training_artifacts,
            select_config_fn=select_anfis_config,
            fixed_config=_build_fixed_config(request),
            fixed_config_kwarg="fixed_anfis_cfg",
        )

    def run_pipeline_native_benchmark(self, request: ANFISBenchmarkRequest):
        return run_sequence_native_benchmark(
            request=request,
            model_name="anfis_only",
            default_output_dir="./functional_api_outputs/anfis_benchmark",
            benchmark_single_dataset_fn=benchmark_single_dataset,
            fixed_config_kwargs={"fixed_anfis_cfg": _build_fixed_config(request)},
        )


ADAPTER = ANFISPipelineAdapter()
