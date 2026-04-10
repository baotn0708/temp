from __future__ import annotations

from functional_api import LSTMTrainRequest, train_lstm


def main() -> None:
    request = LSTMTrainRequest(
        datasets=["AMZN"],
        split_ratio="7/2/1",
        max_epochs=5,
        patience=2,
        batch_size=64,
    )
    result = train_lstm(request)
    for item in result.datasets:
        print(item.dataset, item.payload["test_metrics"]["by_target"]["close"]["rmse"])


if __name__ == "__main__":
    main()
