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

## Đổi dữ liệu đầu vào

Các file trong `examples/` là runner mỏng. Muốn chạy với dữ liệu khác, bạn chỉ cần sửa `datasets=[...]` hoặc `files=[...]` trong file runner tương ứng.

Ví dụ trong một runner train:

```python
request = GRUTrainRequest(
    datasets=["AMZN"],
    ...
)
```

Bạn có thể đổi sang nhiều dataset theo tên:

```python
request = GRUTrainRequest(
    datasets=["AMZN", "JPM", "TSLA"],
    ...
)
```

Hoặc chỉ định trực tiếp file CSV:

```python
request = GRUTrainRequest(
    files=[
        "/absolute/path/to/AMZN.csv",
        "/absolute/path/to/JPM.csv",
    ],
    ...
)
```

Các runner thường cần sửa:

- `examples/run_gru_train.py`
- `examples/run_gru_benchmark.py`
- `examples/run_lstm_train.py`
- `examples/run_lstm_benchmark.py`
- `examples/run_anfis_train.py`
- `examples/run_anfis_benchmark.py`
- `examples/run_feature_group_train.py`
- `examples/run_feature_group_benchmark.py`
- `examples/run_regime_hybrid_train.py`
- `examples/run_regime_hybrid_benchmark.py`
- `examples/run_fair_benchmark.py`

Khuyến nghị:

- Dùng `files=[...]` nếu bạn muốn chỉ rõ chính xác CSV nào sẽ được chạy.
- Dùng `datasets=[...]` nếu bạn đang theo naming convention sẵn có của repo.
- Mỗi file CSV hiện được xử lý như một dataset riêng: train riêng, benchmark riêng, rồi mới tổng hợp bảng kết quả.
