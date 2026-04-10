from __future__ import annotations

from pathlib import Path

import pandas as pd

from functional_api.helpers import (
    FAIR_PROTOCOL_VERSION,
    FAIR_TARGET_TASK,
    aggregate_numeric_rows,
    build_fair_report,
    build_model_ranking,
    ensure_output_dir,
    maybe_write_csv,
    maybe_write_json,
    maybe_write_markdown,
    normalize_model_names,
    validate_adapter_contract,
)
from functional_api.helpers.fairness_helpers import fairness_notes_payload
from functional_api.pipelines import get_adapter
from functional_api.types import FairBenchmarkConfig, FairBenchmarkResult


def benchmark_fair(models: list[str] | tuple[str, ...] | None, request: FairBenchmarkConfig) -> FairBenchmarkResult:
    normalized_names = normalize_model_names(models)
    adapters = [get_adapter(name) for name in normalized_names]
    for adapter in adapters:
        validate_adapter_contract(adapter)

    all_rows: list[dict] = []
    adapter_metadata: dict[str, object] = {}
    for adapter in adapters:
        prepared_items = adapter.prepare_for_fair_benchmark(request)
        dataset_runs: dict[str, object] = {}
        for prepared in prepared_items:
            seed_payloads = {}
            for seed in request.eval_seeds:
                payload = adapter.run_fair_train_eval(prepared, int(seed), request)
                all_rows.extend(payload["rows"])
                seed_payloads[str(int(seed))] = payload["metadata"]
            dataset_runs[getattr(prepared, "dataset_name", "<unknown>")] = seed_payloads
        adapter_metadata[adapter.name] = dataset_runs

    rows_df = pd.DataFrame(all_rows)
    aggregate_df = aggregate_numeric_rows(rows_df, group_cols=["dataset", "model", "split_ratio", "target"])
    ranking_df = build_model_ranking(rows_df)
    metadata = {
        "protocol_version": FAIR_PROTOCOL_VERSION,
        "target_task": FAIR_TARGET_TASK,
        "split_ratio": request.split_ratio,
        "look_back": request.look_back,
        "seq_len": request.seq_len,
        "horizon": request.horizon,
        "gap": request.gap,
        "eval_seeds": [int(seed) for seed in request.eval_seeds],
        "budget_note": (
            "This benchmark equalizes data protocol, split, seed list, and reporting. "
            "It does not enforce equal tuning/search compute across models."
        ),
        "adapter_notes": fairness_notes_payload(adapters),
        "adapter_runs": adapter_metadata,
    }
    report = build_fair_report(rows_df, aggregate_df, ranking_df, metadata)

    output_paths: dict[str, Path] = {}
    if request.artifact_policy.save:
        root = ensure_output_dir(request.artifact_policy, "./functional_api_outputs/fair_benchmark")
        for key, path in {
            "rows": maybe_write_csv(request.artifact_policy, root / "fair_benchmark_rows.csv", rows_df),
            "aggregate": maybe_write_csv(request.artifact_policy, root / "fair_benchmark_aggregate.csv", aggregate_df),
            "ranking": maybe_write_csv(request.artifact_policy, root / "fair_benchmark_ranking.csv", ranking_df),
            "metadata": maybe_write_json(request.artifact_policy, root / "fair_benchmark_metadata.json", metadata),
            "report": maybe_write_markdown(request.artifact_policy, root / "fair_benchmark_report.md", report),
        }.items():
            if path is not None:
                output_paths[key] = path

    return FairBenchmarkResult(
        rows=rows_df,
        aggregate=aggregate_df,
        ranking=ranking_df,
        metadata=metadata,
        output_paths=output_paths,
        report_markdown=report,
    )
