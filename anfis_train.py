from __future__ import annotations

import argparse
from pathlib import Path

from anfis_only_core import ANFISConfig, fit_final_anfis, save_training_artifacts, select_anfis_config
from gru_only_core import RatioSplitConfig, TARGET_KEYS, gather_split_arrays, load_asset_dataset, resolve_datasets
from hybrid_core import TrainConfig


def format_console_metrics(metrics_bundle: dict) -> str:
    parts = []
    by_target = metrics_bundle["by_target"]
    for target in TARGET_KEYS:
        parts.append(f"{target}_rmse={by_target[target]['rmse']:.6f}")
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final ANFIS-only models and evaluate reconstructed next-day OHLC prices."
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
    parser.add_argument("--tuning-seed", type=int, default=123)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--fast", action="store_true", help="Use only the first two ANFIS configs when validation exists.")
    parser.add_argument("--n-rules", type=int, default=None, help="Force a single ANFIS config instead of tuning.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="./anfis_train_outputs")
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

    fixed_anfis_cfg = None
    if args.n_rules is not None:
        fixed_anfis_cfg = ANFISConfig(n_rules=args.n_rules, lr=args.lr)

    datasets = resolve_datasets(args.datasets, args.files)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name, path in datasets:
        asset = load_asset_dataset(path=path, name=dataset_name, split_cfg=split_cfg)
        split = gather_split_arrays(asset, split_cfg)

        anfis_cfg, selected_epoch, val_metrics = select_anfis_config(
            split=split,
            train_cfg=train_cfg,
            tuning_seed=args.tuning_seed,
            fast=args.fast,
            fixed_anfis_cfg=fixed_anfis_cfg,
        )

        model, scaler, test_metrics = fit_final_anfis(
            split=split,
            anfis_cfg=anfis_cfg,
            train_cfg=train_cfg,
            seed=args.train_seed,
            epochs=selected_epoch,
        )

        dataset_output = output_root / dataset_name.lower()
        save_training_artifacts(
            output_dir=dataset_output,
            dataset_name=dataset_name,
            split_cfg=split_cfg,
            train_cfg=train_cfg,
            anfis_cfg=anfis_cfg,
            selected_epoch=selected_epoch,
            sample_counts=split.sample_counts(),
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            scaler=scaler,
            model=model,
        )

        print(
            f"[{dataset_name}] saved to {dataset_output} "
            f"{format_console_metrics(test_metrics)}"
        )


if __name__ == "__main__":
    main()
