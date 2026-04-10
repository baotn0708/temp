from __future__ import annotations

from functional_api import ArtifactPolicy, FairBenchmarkConfig, benchmark_fair


def main() -> None:
    config = FairBenchmarkConfig(
        datasets=["AMZN", "JPM", "TSLA"],
        split_ratio="7/2/1",
        seq_len=60,
        look_back=60,
        horizon=1,
        gap=1,
        eval_seeds=(7,),
        max_epochs=5,
        patience=2,
        batch_size=64,
        artifact_policy=ArtifactPolicy(output_dir="./functional_api_outputs/fair_benchmark_demo", save=True),
    )
    result = benchmark_fair(
        models=["gru", "lstm", "anfis", "feature_group", "regime_hybrid"],
        request=config,
    )
    print(result.ranking.to_string(index=False))


if __name__ == "__main__":
    main()

