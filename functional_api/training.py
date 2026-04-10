from __future__ import annotations

from functional_api.pipelines import get_adapter


def train_gru(request):
    return get_adapter("gru").run_pipeline_native_train(request)


def train_lstm(request):
    return get_adapter("lstm").run_pipeline_native_train(request)


def train_anfis(request):
    return get_adapter("anfis").run_pipeline_native_train(request)


def train_feature_group(request):
    return get_adapter("feature_group").run_pipeline_native_train(request)


def train_regime_hybrid(request):
    return get_adapter("regime_hybrid").run_pipeline_native_train(request)
