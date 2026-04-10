from __future__ import annotations

from functional_api.fair_benchmark import benchmark_fair
from functional_api.pipelines import get_adapter, get_available_models


def benchmark_gru(request):
    return get_adapter("gru").run_pipeline_native_benchmark(request)


def benchmark_lstm(request):
    return get_adapter("lstm").run_pipeline_native_benchmark(request)


def benchmark_anfis(request):
    return get_adapter("anfis").run_pipeline_native_benchmark(request)


def benchmark_feature_group(request):
    return get_adapter("feature_group").run_pipeline_native_benchmark(request)


def benchmark_regime_hybrid(request):
    return get_adapter("regime_hybrid").run_pipeline_native_benchmark(request)


def benchmark_pipeline_native(model: str, request):
    normalized = model.strip().lower()
    if normalized not in get_available_models():
        raise ValueError(f"Unsupported pipeline model: {normalized}")
    return get_adapter(normalized).run_pipeline_native_benchmark(request)
