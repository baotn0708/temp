from __future__ import annotations

from functional_api import ArtifactPolicy, RegimeHybridBenchmarkRequest, benchmark_regime_hybrid


def main() -> None:
    request = RegimeHybridBenchmarkRequest(
        datasets=["AMZN"],
        look_back=60,
        n_splits=1,
        epochs=5,
        batch_size=32,
        eval_seeds=(7,),
        artifact_policy=ArtifactPolicy(save=False),
    )
    result = benchmark_regime_hybrid(request)
    print(result.rows.to_string(index=False))


if __name__ == "__main__":
    main()
