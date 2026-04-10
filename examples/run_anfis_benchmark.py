from __future__ import annotations

from functional_api import ANFISBenchmarkRequest, ArtifactPolicy, benchmark_anfis


def main() -> None:
    request = ANFISBenchmarkRequest(
        datasets=["AMZN"],
        split_ratio="7/2/1",
        max_epochs=5,
        patience=2,
        batch_size=64,
        eval_seeds=(7,),
        artifact_policy=ArtifactPolicy(save=False),
    )
    result = benchmark_anfis(request)
    print(result.rows.to_string(index=False))


if __name__ == "__main__":
    main()
