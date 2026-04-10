from __future__ import annotations

from pathlib import Path
from typing import Sequence

from functional_api.core.gru import resolve_datasets

from functional_api.types import DatasetSpec


def resolve_dataset_specs(
    datasets: Sequence[str] | None,
    files: Sequence[str] | None,
) -> list[DatasetSpec]:
    return [
        DatasetSpec(name=name, path=Path(path).resolve())
        for name, path in resolve_datasets(dataset_names=datasets, files=files)
    ]


def dataset_specs_to_serializable(specs: Sequence[DatasetSpec]) -> list[dict[str, str]]:
    return [{"name": spec.name, "path": str(spec.path)} for spec in specs]

