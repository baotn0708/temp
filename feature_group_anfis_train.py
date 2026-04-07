from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        description="Train final feature-group ANFIS on a chronological 7/3 split."
    )
    parser.add_argument("--datasets", nargs="+", default=["AMZN", "JPM", "TSLA"])
    parser.add_argument("--files", nargs="*", default=None, help="Optional CSV paths. If set, --datasets is ignored.")
    parser.add_argument("--split-ratio", choices=["7/3"], default="7/3")
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--n-mfs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--include-exog", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./feature_group_anfis_train_outputs")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> FeatureGroupAnfis703Config:
    return FeatureGroupAnfis703Config(
        look_back=args.look_back,
        split_ratio=args.split_ratio,
        gap=args.gap,
        n_mfs=args.n_mfs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        include_exog=args.include_exog,
        max_rows=args.max_rows,
        verbose=args.verbose,
    )


def format_console_metrics(metrics_flat: dict) -> str:
    return " ".join(
        [
            f"open_rmse={metrics_flat['open_rmse']:.6f}",
            f"high_rmse={metrics_flat['high_rmse']:.6f}",
            f"low_rmse={metrics_flat['low_rmse']:.6f}",
            f"close_rmse={metrics_flat['close_rmse']:.6f}",
        ]
    )


def main() -> None:
    configure_runtime()
    args = parse_args()
    set_seed(args.seed)
    datasets = resolve_datasets(args.datasets, args.files)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(args)

    for dataset_name, path in datasets:
        result = run_feature_group_anfis_703(path, dataset_name, cfg)
        dataset_output = output_root / dataset_name.lower()
        dataset_output.mkdir(parents=True, exist_ok=True)

        model_path = dataset_output / "feature_group_anfis_clean.keras"
        result["model"].save(model_path)

        metrics_flat = flatten_price_metrics(result["test_metrics"])
        summary = {
            "dataset": dataset_name,
            "model": MODEL_NAME,
            "protocol": {
                "split_ratio": args.split_ratio,
                "look_back": args.look_back,
                "gap": args.gap,
                "forecast_horizon": "next trading day",
                "validation": "not used",
                "selection_rule": "fixed config only; test evaluated once after training",
            },
            "sample_counts": {
                "train": int(len(result["prepared"].y_train)),
                "test": int(len(result["prepared"].y_test)),
            },
            "model_config": {
                "n_mfs": args.n_mfs,
                "lstm_units": args.lstm_units,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
            "train_time_sec": round(float(result["train_time"]), 2),
            "epochs_trained": int(len(result["history"]["loss"])),
            "test_metrics": result["test_metrics"],
            "test_metrics_flat": metrics_flat,
        }

        (dataset_output / "training_summary.json").write_text(
            json.dumps(to_jsonable(summary), indent=2),
            encoding="utf-8",
        )
        (dataset_output / "rules.json").write_text(
            json.dumps(to_jsonable(result["rules"]), indent=2),
            encoding="utf-8",
        )
        (dataset_output / "sample_analysis.json").write_text(
            json.dumps(to_jsonable(result["sample_analysis"]), indent=2),
            encoding="utf-8",
        )
        (dataset_output / "history.json").write_text(
            json.dumps(to_jsonable({"history": result["history"]}), indent=2),
            encoding="utf-8",
        )

        print(f"[{dataset_name}] saved to {dataset_output} {format_console_metrics(metrics_flat)}")


if __name__ == "__main__":
    main()
