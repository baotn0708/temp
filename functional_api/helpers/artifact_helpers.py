from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from functional_api.types import ArtifactPolicy


def ensure_output_dir(policy: ArtifactPolicy, default_dir: str | Path) -> Path:
    root = Path(policy.output_dir or default_dir).resolve()
    if policy.save:
        root.mkdir(parents=True, exist_ok=True)
    return root


def maybe_write_json(policy: ArtifactPolicy, path: Path, payload: Any) -> Path | None:
    if not policy.save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def maybe_write_markdown(policy: ArtifactPolicy, path: Path, content: str) -> Path | None:
    if not policy.save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def maybe_write_csv(policy: ArtifactPolicy, path: Path, frame: pd.DataFrame) -> Path | None:
    if not policy.save:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
