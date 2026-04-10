# Functional API

Repo này đã được rút gọn về một đường chạy chính duy nhất: `functional_api`.

## Cấu trúc

- `functional_api/`: public API để train và benchmark
- `functional_api/core/`: phần lõi private phía sau public API
- `examples/`: script mẫu để bấm `Run` trong editor hoặc chạy trực tiếp bằng Python

## Cài môi trường

```bash
python3 -m venv .venv-functional-api
source .venv-functional-api/bin/activate
python -m pip install -r functional_api/requirements.txt
```

## Cách dùng

Train một pipeline:

```python
from functional_api import GRUTrainRequest, train_gru

result = train_gru(
    GRUTrainRequest(
        files=["/path/to/data.csv"],
    )
)
```

Benchmark công bằng nhiều model:

```python
from functional_api import ArtifactPolicy, FairBenchmarkConfig, benchmark_fair

result = benchmark_fair(
    models=["gru", "lstm", "anfis", "feature_group", "regime_hybrid"],
    request=FairBenchmarkConfig(
        files=["/path/to/data.csv"],
        artifact_policy=ArtifactPolicy(save=False),
    ),
)
```

## Script mẫu

```bash
python examples/run_gru_benchmark.py
python examples/run_gru_train.py
python examples/run_lstm_benchmark.py
python examples/run_lstm_train.py
python examples/run_anfis_benchmark.py
python examples/run_anfis_train.py
python examples/run_feature_group_benchmark.py
python examples/run_feature_group_train.py
python examples/run_regime_hybrid_benchmark.py
python examples/run_regime_hybrid_train.py
python examples/run_fair_benchmark.py
```
