from __future__ import annotations

from functional_api import RegimeHybridTrainRequest, train_regime_hybrid


def main() -> None:
    request = RegimeHybridTrainRequest(
        datasets=["AMZN"],
        look_back=60,
        epochs=5,
        batch_size=32,
        runs=1,
    )
    result = train_regime_hybrid(request)
    for item in result.datasets:
        print(item.dataset, item.payload["best_run_seed"])


if __name__ == "__main__":
    main()
