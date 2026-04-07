from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from feature_group_anfis_703_core import (
    MODEL_NAME,
    FeatureGroupAnfis703Config,
    flatten_price_metrics,
    run_feature_group_anfis_703,
)
from gru_only_core import resolve_datasets
from run_feature_group_anfis_clean import configure_runtime, set_seed, to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean benchmark for feature-group ANFIS on a chronological 7/3 split."
    )
    parser.add_argument("--datasets", nargs="+", default=["AMZN", "JPM", "TSLA"])
    parser.add_argument("--files", nargs="*", default=None, help="Optional CSV paths. If set, --datasets is ignored.")
    parser.add_argument("--split-ratio", choices=["7/3"], default="7/3")
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--n-mfs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--include-exog", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./feature_group_anfis_benchmark_outputs")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace, seed: int) -> FeatureGroupAnfis703Config:
    return FeatureGroupAnfis703Config(
        look_back=args.look_back,
        split_ratio=args.split_ratio,
        gap=args.gap,
        n_mfs=args.n_mfs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        include_exog=args.include_exog,
        max_rows=args.max_rows,
        verbose=args.verbose,
    )


def aggregate_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in rows_df.columns if col not in {"dataset", "seed", "model", "split_ratio"}]
    grouped = rows_df.groupby(["dataset", "model", "split_ratio"], as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [
        col if isinstance(col, str) else f"{col[0]}_{'avg_over_seeds' if col[1] == 'mean' else 'std_over_seeds'}"
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def format_console_metrics(row: Dict[str, float]) -> str:
    return " ".join(
        [
            f"open_rmse={row['open_rmse']:.6f}",
            f"high_rmse={row['high_rmse']:.6f}",
            f"low_rmse={row['low_rmse']:.6f}",
            f"close_rmse={row['close_rmse']:.6f}",
        ]
    )


def main() -> None:
    configure_runtime()
    args = parse_args()
    datasets = resolve_datasets(args.datasets, args.files)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []

    for dataset_name, path in datasets:
        dataset_output = output_root / dataset_name.lower()
        dataset_output.mkdir(parents=True, exist_ok=True)
        rows: List[dict] = []
        runs_meta: Dict[str, dict] = {}

        for seed in args.eval_seeds:
            set_seed(int(seed))
            cfg = build_cfg(args, seed=int(seed))
            result = run_feature_group_anfis_703(path, dataset_name, cfg)
            metrics_flat = flatten_price_metrics(result["test_metrics"])
            row = {
                "dataset": dataset_name,
                "seed": int(seed),
                "model": MODEL_NAME,
                "split_ratio": args.split_ratio,
                **metrics_flat,
                "epochs_trained": int(len(result["history"]["loss"])),
                "train_time_sec": round(float(result["train_time"]), 2),
                "train_samples": int(len(result["prepared"].y_train)),
                "test_samples": int(len(result["prepared"].y_test)),
            }
            rows.append(row)
            runs_meta[str(int(seed))] = {
                "metrics": result["test_metrics"],
                "sample_counts": {
                    "train": int(len(result["prepared"].y_train)),
                    "test": int(len(result["prepared"].y_test)),
                },
                "epochs_trained": int(len(result["history"]["loss"])),
                "train_time_sec": round(float(result["train_time"]), 2),
            }

        rows_df = pd.DataFrame(rows)
        aggregate_df = aggregate_rows(rows_df) if len(rows_df) > 1 else pd.DataFrame()

        rows_df.to_csv(dataset_output / "benchmark_results.csv", index=False)
        if not aggregate_df.empty:
            aggregate_df.to_csv(dataset_output / "benchmark_aggregate.csv", index=False)

        metadata = {
            "dataset": dataset_name,
            "path": str(path),
            "protocol": {
                "split_ratio": args.split_ratio,
                "look_back": args.look_back,
                "gap": args.gap,
                "forecast_horizon": "next trading day",
                "selection_rule": "no validation; fixed config only; test evaluated once per run",
            },
            "model_config": {
                "n_mfs": args.n_mfs,
                "lstm_units": args.lstm_units,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
            "eval_seeds": [int(seed) for seed in args.eval_seeds],
            "runs": runs_meta,
        }
        (dataset_output / "benchmark_metadata.json").write_text(
            json.dumps(to_jsonable(metadata), indent=2),
            encoding="utf-8",
        )

        report_lines = [
            f"# Feature-group ANFIS benchmark: {dataset_name}",
            "",
            "## Protocol",
            f"- split_ratio = {args.split_ratio}",
            f"- lookback_days = {args.look_back}",
            "- forecast = next-day OHLC via original target parameterization in run_feature_group_anfis_clean.py",
            f"- gap = {args.gap}",
            "- validation = not used",
            "- model selection = none; fixed config benchmark only",
            f"- eval_seeds = {args.eval_seeds}",
            "",
            "## Raw benchmark rows",
            "",
            rows_df.to_markdown(index=False),
        ]
        if not aggregate_df.empty:
            report_lines.extend(
                [
                    "",
                    "## Aggregate over seeds",
                    "",
                    aggregate_df.to_markdown(index=False),
                ]
            )
        (dataset_output / "benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")

        all_rows.extend(rows)
        print(f"[{dataset_name}] split={args.split_ratio} {format_console_metrics(rows[0])}")

    combined_results = pd.DataFrame(all_rows)
    combined_results.to_csv(output_root / "combined_benchmark_results.csv", index=False)
    if len(combined_results) > 1:
        combined_aggregate = aggregate_rows(combined_results)
        combined_aggregate.to_csv(output_root / "combined_benchmark_aggregate.csv", index=False)
    else:
        combined_aggregate = pd.DataFrame()

    report_lines = [
        "# Combined feature-group ANFIS benchmark",
        "",
        "## Raw benchmark rows",
        "",
        combined_results.to_markdown(index=False),
    ]
    if not combined_aggregate.empty:
        report_lines.extend(
            [
                "",
                "## Aggregate over seeds",
                "",
                combined_aggregate.to_markdown(index=False),
            ]
        )
    (output_root / "combined_benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nCombined results:")
    print(combined_results.to_string(index=False))


if __name__ == "__main__":
    main()
