from __future__ import annotations

from typing import Iterable, Protocol, Sequence


FAIR_PROTOCOL_VERSION = "fair_v1_chronological_purged_ratio"
FAIR_TARGET_TASK = "next-day OHLC forecasting benchmarked on reconstructed OHLC price metrics"

SUPPORTED_FAIR_MODELS = (
    "gru",
    "lstm",
    "anfis",
    "feature_group",
    "regime_hybrid",
)


class PipelineAdapter(Protocol):
    name: str
    benchmark_label: str
    fairness_note: str

    def prepare_for_fair_benchmark(self, request): ...

    def run_fair_train_eval(self, prepared, seed: int, request): ...

    def run_pipeline_native_train(self, request): ...

    def run_pipeline_native_benchmark(self, request): ...


def normalize_model_names(models: Sequence[str] | None) -> list[str]:
    requested = list(models or SUPPORTED_FAIR_MODELS)
    normalized = [item.strip().lower() for item in requested]
    invalid = sorted(set(normalized) - set(SUPPORTED_FAIR_MODELS))
    if invalid:
        raise ValueError(f"Unsupported fair benchmark models: {invalid}")
    return normalized


def validate_adapter_contract(adapter: PipelineAdapter) -> None:
    required = [
        "prepare_for_fair_benchmark",
        "run_fair_train_eval",
        "run_pipeline_native_train",
        "run_pipeline_native_benchmark",
    ]
    missing = [name for name in required if not hasattr(adapter, name)]
    if missing:
        raise TypeError(f"Adapter '{getattr(adapter, 'name', '<unknown>')}' is missing methods: {missing}")


def fairness_notes_payload(adapters: Iterable[PipelineAdapter]) -> dict[str, str]:
    return {adapter.name: adapter.fairness_note for adapter in adapters}

