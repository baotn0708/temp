from __future__ import annotations

from functional_api import FeatureGroupTrainRequest, train_feature_group


def main() -> None:
    request = FeatureGroupTrainRequest(
        datasets=["AMZN"],
        split_ratio="7/3",
        look_back=60,
        epochs=5,
        batch_size=32,
    )
    result = train_feature_group(request)
    for item in result.datasets:
        print(item.dataset, item.payload["test_metrics_flat"]["close_rmse"])


if __name__ == "__main__":
    main()
