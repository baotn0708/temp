from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

from gru_only_core import resolve_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the regime-gated ANFIS hybrid model on AMZN/JPM/TSLA."
    )
    parser.add_argument("--datasets", nargs="+", default=["AMZN", "JPM", "TSLA"])
    parser.add_argument("--files", nargs="*", default=None, help="Optional CSV paths. If set, --datasets is ignored.")
    parser.add_argument("--output-dir", type=str, default="./regime_gated_anfis_hybrid_outputs")
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--n-mfs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-units", type=int, default=48)
    parser.add_argument("--conv-filters", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--component-loss-weight", type=float, default=0.25)
    parser.add_argument("--include-exog", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def build_run_args(args: argparse.Namespace, stock_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        data=".",
        output_dir=args.output_dir,
        stock_name=stock_name,
        look_back=args.look_back,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        n_mfs=args.n_mfs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        runs=args.runs,
        seed=args.seed,
        temporal_units=args.temporal_units,
        conv_filters=args.conv_filters,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        component_loss_weight=args.component_loss_weight,
        include_exog=args.include_exog,
        max_rows=args.max_rows,
        upload_on_colab=False,
        verbose=args.verbose,
    )


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args.datasets, args.files)

    try:
        import keras
        from run_regime_gated_anfis_hybrid import (
            print_overall_summary,
            print_summary,
            run_training_for_path,
            save_artifacts,
            save_overall_summary,
            summarize_result_for_overview,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to import run_regime_gated_anfis_hybrid.py. "
            "Check the Python environment first, especially that NumPy matches requirements.txt "
            "(this project expects numpy<2)."
        ) from exc

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    overall_summaries = []
    with keras.device("cpu"):
        for dataset_name, path in datasets:
            run_args = build_run_args(args=args, stock_name=dataset_name)
            results = run_training_for_path(run_args, Path(path))
            artifact_dir = save_artifacts(run_args, results)
            print_summary(results, artifact_dir)
            overall_summaries.append(summarize_result_for_overview(results, artifact_dir))

    summary_path = save_overall_summary(args, overall_summaries)
    if len(overall_summaries) > 1:
        print_overall_summary(overall_summaries, summary_path)


if __name__ == "__main__":
    main()
