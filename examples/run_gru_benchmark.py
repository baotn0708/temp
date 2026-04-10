from __future__ import annotations

from functional_api import ArtifactPolicy, GRUBenchmarkRequest, benchmark_gru


def main() -> None:
    request = GRUBenchmarkRequest(
        datasets=["AMZN"],
        split_ratio="7/2/1",
        max_epochs=5,
        patience=2,
        batch_size=64,
        eval_seeds=(7,),
        artifact_policy=ArtifactPolicy(save=False),
    )
    result = benchmark_gru(request)
    print(result.rows.to_string(index=False))


if __name__ == "__main__":
    main()
