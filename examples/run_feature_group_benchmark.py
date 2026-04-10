from __future__ import annotations

from functional_api import ArtifactPolicy, FeatureGroupBenchmarkRequest, benchmark_feature_group


def main() -> None:
    request = FeatureGroupBenchmarkRequest(
        datasets=["AMZN"],
        split_ratio="7/3",
        look_back=60,
        epochs=5,
        batch_size=32,
        eval_seeds=(7,),
        artifact_policy=ArtifactPolicy(save=False),
    )
    result = benchmark_feature_group(request)
    print(result.rows.to_string(index=False))


if __name__ == "__main__":
    main()
