from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

from gru_only_core import (
    GRUConfig,
    RatioSplitConfig,
    TARGET_KEYS,
    benchmark_single_dataset,
    resolve_datasets,
)
from hybrid_core import TrainConfig, save_json


def make_wide_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.pivot_table(
        index=["dataset", "seed", "model", "split_ratio"],
        columns="target",
        values=["rmse", "mape", "mae", "r2", "sign_acc"],
        aggfunc="first",
    ).reset_index()
    summary.columns = [
        f"{target}_{metric}" if target else metric
        for metric, target in summary.columns.to_flat_index()
    ]
    return summary


def format_console_metrics(rows: pd.DataFrame, seed: int) -> str:
    parts = []
    seed_rows = rows[rows["seed"] == seed]
    for target in TARGET_KEYS:
        row = seed_rows[seed_rows["target"] == target].iloc[0]
        parts.append(f"{target}_rmse={row['rmse']:.6f}")
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean GRU-only benchmark for AMZN/JPM/TSLA with chronological 7/3 or 7/2/1 splits."
    )
    parser.add_argument("--datasets", nargs="+", default=["AMZN", "JPM", "TSLA"])
    parser.add_argument("--files", nargs="*", default=None, help="Optional CSV paths. If set, --datasets is ignored.")
    parser.add_argument("--split-ratio", choices=["7/3", "7/2/1"], default="7/2/1")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--gap", type=int, default=-1, help="Use -1 to default gap=horizon.")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--tuning-seed", type=int, default=123)
    parser.add_argument("--fast", action="store_true", help="Use only the first two GRU configs when validation exists.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Force a single GRU config instead of tuning.")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="./gru_benchmark_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_cfg = RatioSplitConfig(
        split_ratio=args.split_ratio,
        seq_len=args.seq_len,
        horizon=args.horizon,
        gap=None if args.gap < 0 else args.gap,
    )
    train_cfg = TrainConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    fixed_gru_cfg = None
    if args.hidden_dim is not None:
        fixed_gru_cfg = GRUConfig(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
        )

    datasets = resolve_datasets(args.datasets, args.files)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    dataset_summaries: List[pd.DataFrame] = []

    for dataset_name, path in datasets:
        dataset_output = output_root / dataset_name.lower()
        dataset_output.mkdir(parents=True, exist_ok=True)

        result = benchmark_single_dataset(
            dataset_name=dataset_name,
            path=path,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
            eval_seeds=args.eval_seeds,
            tuning_seed=args.tuning_seed,
            fast=args.fast,
            fixed_gru_cfg=fixed_gru_cfg,
        )

        rows = pd.DataFrame(result["rows"])
        rows.to_csv(dataset_output / "benchmark_results.csv", index=False)
        all_rows.extend(result["rows"])

        summary = make_wide_summary(rows)
        summary.to_csv(dataset_output / "benchmark_summary.csv", index=False)
        dataset_summaries.append(summary)

        save_json(
            dataset_output / "benchmark_metadata.json",
            {
                "dataset": dataset_name,
                "path": result["path"],
                "split_config": asdict(split_cfg),
                "train_config": asdict(train_cfg),
                "sample_counts": result["sample_counts"],
                "selected_config": result["selected_config"],
                "selected_epoch": result["selected_epoch"],
                "val_metrics": result["val_metrics"],
                "metrics_by_seed": result["metrics_by_seed"],
            },
        )
        save_json(
            dataset_output / "selected_config.json",
            {
                "dataset": dataset_name,
                "model": "gru_only",
                "split_ratio": split_cfg.split_ratio,
                "target_definition": "model predicts next-day log-return for Open, High, Low, Close",
                "evaluation_definition": "benchmark metrics are computed on algebraically reconstructed next-day OHLC prices",
                "selected_config": result["selected_config"],
                "selected_epoch": result["selected_epoch"],
                "val_metrics": result["val_metrics"],
            },
        )

        report_lines = [
            f"# GRU-only benchmark: {dataset_name}",
            "",
            "## Protocol",
            f"- split_ratio = {split_cfg.split_ratio}",
            f"- lookback_days = {split_cfg.seq_len}",
            f"- forecast_horizon_days = {split_cfg.horizon}",
            f"- gap = {split_cfg.effective_gap}",
            "- target = model predicts next-day log-return for each of Open, High, Low, Close",
            "- evaluation = reconstruct next-day OHLC algebraically, then compare directly with test OHLC prices",
            f"- eval_seeds = {args.eval_seeds}",
            f"- sample_counts = {result['sample_counts']}",
            "",
            "## Selected config",
            f"- config = {result['selected_config']}",
            f"- selected_epoch = {result['selected_epoch']}",
            f"- val_metrics = {result['val_metrics']}",
            "",
            "## Benchmark rows",
            "",
            rows.to_markdown(index=False),
            "",
            "## Wide summary",
            "",
            summary.to_markdown(index=False),
        ]
        (dataset_output / "benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")

        first_seed = int(rows["seed"].iloc[0])
        print(f"[{dataset_name}] split={split_cfg.split_ratio} {format_console_metrics(rows, first_seed)}")

    combined_results = pd.DataFrame(all_rows)
    combined_results.to_csv(output_root / "combined_benchmark_results.csv", index=False)

    combined_summary = pd.concat(dataset_summaries, ignore_index=True)
    combined_summary.to_csv(output_root / "combined_benchmark_summary.csv", index=False)

    combined_report_lines = [
        "# Combined GRU-only benchmark",
        "",
        "## Protocol",
        f"- split_ratio = {split_cfg.split_ratio}",
        f"- lookback_days = {split_cfg.seq_len}",
        f"- forecast_horizon_days = {split_cfg.horizon}",
        f"- gap = {split_cfg.effective_gap}",
        "- target = model predicts next-day log-return for each of Open, High, Low, Close",
        "- evaluation = reconstruct next-day OHLC algebraically, then compare directly with test OHLC prices",
        f"- eval_seeds = {args.eval_seeds}",
        "",
        "## Benchmark rows",
        "",
        combined_results.to_markdown(index=False),
        "",
        "## Wide summary",
        "",
        combined_summary.to_markdown(index=False),
    ]
    (output_root / "combined_benchmark_report.md").write_text("\n".join(combined_report_lines), encoding="utf-8")

    print("\nCombined summary:")
    print(combined_summary.to_string(index=False))


if __name__ == "__main__":
    main()
