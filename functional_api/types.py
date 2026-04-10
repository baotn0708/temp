from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


@dataclass
class ArtifactPolicy:
    output_dir: str | Path | None = None
    save: bool = True


@dataclass
class TrainRequestBase:
    datasets: Sequence[str] | None = None
    files: Sequence[str] | None = None
    artifact_policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)


@dataclass
class BenchmarkRequestBase:
    datasets: Sequence[str] | None = None
    files: Sequence[str] | None = None
    eval_seeds: Sequence[int] = (7, 21, 42)
    artifact_policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)


@dataclass
class FairBenchmarkConfig(BenchmarkRequestBase):
    split_ratio: str = "7/2/1"
    seq_len: int = 60
    look_back: int = 60
    horizon: int = 1
    gap: int = 1
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    tuning_seed: int = 123
    max_rows: int | None = None
    include_exog: bool = False
    fast: bool = False
    model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TrainDatasetResult:
    dataset: str
    payload: Mapping[str, Any]
    artifact_dir: Path | None = None


@dataclass
class TrainPipelineResult:
    model: str
    datasets: list[TrainDatasetResult]
    metadata: Mapping[str, Any]
    output_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass
class BenchmarkPipelineResult:
    model: str
    rows: pd.DataFrame
    aggregate: pd.DataFrame
    metadata: Mapping[str, Any]
    output_paths: Mapping[str, Path] = field(default_factory=dict)
    report_markdown: Optional[str] = None


@dataclass
class FairBenchmarkResult:
    rows: pd.DataFrame
    aggregate: pd.DataFrame
    ranking: pd.DataFrame
    metadata: Mapping[str, Any]
    output_paths: Mapping[str, Path] = field(default_factory=dict)
    report_markdown: Optional[str] = None

