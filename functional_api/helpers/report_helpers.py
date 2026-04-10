from __future__ import annotations

import json
from typing import Mapping, Sequence

import pandas as pd

PRICE_NAME_TO_TARGET = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
}


def aggregate_numeric_rows(rows_df: pd.DataFrame, group_cols: Sequence[str], skip_cols: Sequence[str] = ()) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()
    excluded = set(group_cols) | set(skip_cols)
    metric_cols = [col for col in rows_df.columns if col not in excluded and pd.api.types.is_numeric_dtype(rows_df[col])]
    grouped = rows_df.groupby(list(group_cols), as_index=False)[metric_cols].agg(["mean", "std"])
    renamed: list[str] = []
    for col in grouped.columns.to_flat_index():
        if isinstance(col, str):
            renamed.append(col)
        elif col[1] == "":
            renamed.append(col[0])
        else:
            renamed.append(f"{col[0]}_{'avg_over_runs' if col[1] == 'mean' else 'std_over_runs'}")
    grouped.columns = renamed
    return grouped


def build_model_ranking(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()
    ranking = (
        rows_df.groupby(["dataset", "model"], as_index=False)[["rmse", "mae", "mape", "r2"]]
        .mean()
        .rename(
            columns={
                "rmse": "avg_rmse",
                "mae": "avg_mae",
                "mape": "avg_mape",
                "r2": "avg_r2",
            }
        )
        .sort_values(["dataset", "avg_rmse", "avg_mae", "model"], ignore_index=True)
    )
    ranking["rank_within_dataset"] = ranking.groupby("dataset")["avg_rmse"].rank(method="dense").astype(int)
    return ranking


def price_metrics_to_rows(
    *,
    dataset_name: str,
    seed: int,
    model: str,
    split_ratio: str,
    metrics: Mapping[str, object],
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    price_metrics = metrics["price_metrics"]
    for price_name, target in PRICE_NAME_TO_TARGET.items():
        item = price_metrics[price_name]
        sign_acc = None
        if target == "open":
            sign_acc = float(metrics.get("open_direction_accuracy", float("nan")))
        elif target == "close":
            sign_acc = float(metrics.get("close_direction_accuracy", float("nan")))
        rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "model": model,
                "split_ratio": split_ratio,
                "target": target,
                "rmse": float(item["RMSE"]),
                "mape": float(item["MAPE"]),
                "mae": float(item["MAE"]),
                "r2": float(item["R2"]),
                "sign_acc": sign_acc,
            }
        )
    return rows


def build_native_report(title: str, rows: pd.DataFrame, aggregate: pd.DataFrame, metadata: Mapping[str, object]) -> str:
    lines = [
        f"# {title}",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, default=str),
        "```",
        "",
        "## Raw rows",
        "",
        rows.to_markdown(index=False) if not rows.empty else "(empty)",
        "",
        "## Aggregate",
        "",
        aggregate.to_markdown(index=False) if not aggregate.empty else "(empty)",
    ]
    return "\n".join(lines)


def build_fair_report(
    rows: pd.DataFrame,
    aggregate: pd.DataFrame,
    ranking: pd.DataFrame,
    metadata: Mapping[str, object],
) -> str:
    lines = [
        "# Fair Benchmark Report",
        "",
        "## Protocol",
        "",
        "```json",
        json.dumps(metadata, indent=2, default=str),
        "```",
        "",
        "## Raw rows",
        "",
        rows.to_markdown(index=False) if not rows.empty else "(empty)",
        "",
        "## Aggregate",
        "",
        aggregate.to_markdown(index=False) if not aggregate.empty else "(empty)",
        "",
        "## Ranking",
        "",
        ranking.to_markdown(index=False) if not ranking.empty else "(empty)",
    ]
    return "\n".join(lines)
