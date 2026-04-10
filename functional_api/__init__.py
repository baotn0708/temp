from .fair_benchmark import benchmark_fair
from .types import ArtifactPolicy, FairBenchmarkConfig

__all__ = [
    "ArtifactPolicy",
    "FairBenchmarkConfig",
    "benchmark_fair",
    "benchmark_pipeline_native",
    "benchmark_gru",
    "benchmark_lstm",
    "benchmark_anfis",
    "benchmark_feature_group",
    "benchmark_regime_hybrid",
    "train_gru",
    "train_lstm",
    "train_anfis",
    "train_feature_group",
    "train_regime_hybrid",
    "GRUTrainRequest",
    "GRUBenchmarkRequest",
    "LSTMTrainRequest",
    "LSTMBenchmarkRequest",
    "ANFISTrainRequest",
    "ANFISBenchmarkRequest",
    "FeatureGroupTrainRequest",
    "FeatureGroupBenchmarkRequest",
    "RegimeHybridTrainRequest",
    "RegimeHybridBenchmarkRequest",
]


def train_gru(request):
    from .training import train_gru as _train_gru

    return _train_gru(request)


def train_lstm(request):
    from .training import train_lstm as _train_lstm

    return _train_lstm(request)


def train_anfis(request):
    from .training import train_anfis as _train_anfis

    return _train_anfis(request)


def train_feature_group(request):
    from .training import train_feature_group as _train_feature_group

    return _train_feature_group(request)


def train_regime_hybrid(request):
    from .training import train_regime_hybrid as _train_regime_hybrid

    return _train_regime_hybrid(request)


def benchmark_gru(request):
    from .benchmarking import benchmark_gru as _benchmark_gru

    return _benchmark_gru(request)


def benchmark_lstm(request):
    from .benchmarking import benchmark_lstm as _benchmark_lstm

    return _benchmark_lstm(request)


def benchmark_anfis(request):
    from .benchmarking import benchmark_anfis as _benchmark_anfis

    return _benchmark_anfis(request)


def benchmark_feature_group(request):
    from .benchmarking import benchmark_feature_group as _benchmark_feature_group

    return _benchmark_feature_group(request)


def benchmark_regime_hybrid(request):
    from .benchmarking import benchmark_regime_hybrid as _benchmark_regime_hybrid

    return _benchmark_regime_hybrid(request)


def benchmark_pipeline_native(model: str, request):
    from .benchmarking import benchmark_pipeline_native as _benchmark_pipeline_native

    return _benchmark_pipeline_native(model, request)


def __getattr__(name: str):
    from .pipelines import __getattr__ as _pipelines_getattr

    if name in {
        "GRUTrainRequest",
        "GRUBenchmarkRequest",
        "LSTMTrainRequest",
        "LSTMBenchmarkRequest",
        "ANFISTrainRequest",
        "ANFISBenchmarkRequest",
        "FeatureGroupTrainRequest",
        "FeatureGroupBenchmarkRequest",
        "RegimeHybridTrainRequest",
        "RegimeHybridBenchmarkRequest",
    }:
        return _pipelines_getattr(name)
    raise AttributeError(name)
