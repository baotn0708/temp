from __future__ import annotations

from importlib import import_module


_REQUEST_EXPORTS = {
    "GRUTrainRequest": ("functional_api.requests", "GRUTrainRequest"),
    "GRUBenchmarkRequest": ("functional_api.requests", "GRUBenchmarkRequest"),
    "LSTMTrainRequest": ("functional_api.requests", "LSTMTrainRequest"),
    "LSTMBenchmarkRequest": ("functional_api.requests", "LSTMBenchmarkRequest"),
    "ANFISTrainRequest": ("functional_api.requests", "ANFISTrainRequest"),
    "ANFISBenchmarkRequest": ("functional_api.requests", "ANFISBenchmarkRequest"),
    "FeatureGroupTrainRequest": ("functional_api.requests", "FeatureGroupTrainRequest"),
    "FeatureGroupBenchmarkRequest": ("functional_api.requests", "FeatureGroupBenchmarkRequest"),
    "RegimeHybridTrainRequest": ("functional_api.requests", "RegimeHybridTrainRequest"),
    "RegimeHybridBenchmarkRequest": ("functional_api.requests", "RegimeHybridBenchmarkRequest"),
}

_ADAPTER_EXPORTS = {
    "gru": ("functional_api.pipelines.gru", "ADAPTER"),
    "lstm": ("functional_api.pipelines.lstm", "ADAPTER"),
    "anfis": ("functional_api.pipelines.anfis", "ADAPTER"),
    "feature_group": ("functional_api.pipelines.feature_group", "ADAPTER"),
    "regime_hybrid": ("functional_api.pipelines.regime_hybrid", "ADAPTER"),
}


def get_adapter(name: str):
    normalized = name.strip().lower()
    if normalized not in _ADAPTER_EXPORTS:
        raise ValueError(f"Unsupported adapter: {normalized}")
    module_name, attr_name = _ADAPTER_EXPORTS[normalized]
    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on local ML environment
        raise RuntimeError(
            f"Failed to import adapter '{normalized}'. "
            f"This usually means the model's runtime dependencies are not healthy in the current environment."
        ) from exc
    return getattr(module, attr_name)


def get_available_models() -> list[str]:
    return list(_ADAPTER_EXPORTS)


def __getattr__(name: str):
    if name in _REQUEST_EXPORTS:
        module_name, attr_name = _REQUEST_EXPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(name)


__all__ = ["get_adapter", "get_available_models", *_REQUEST_EXPORTS.keys()]
