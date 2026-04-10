from __future__ import annotations

from functional_api import ArtifactPolicy, LSTMBenchmarkRequest, benchmark_lstm


def main() -> None:
    request = LSTMBenchmarkRequest(
        datasets=["AMZN"],
        split_ratio="7/2/1",
        max_epochs=5,
        patience=2,
        batch_size=64,
        eval_seeds=(7,),
        artifact_policy=ArtifactPolicy(save=False),
    )
    result = benchmark_lstm(request)
    print(result.rows.to_string(index=False))


if __name__ == "__main__":
    main()
