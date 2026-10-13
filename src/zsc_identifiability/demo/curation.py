"""Load and validate the deterministic public replay selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from zsc_identifiability.demo.models import DemoCuration


def load_demo_curation(path: str | Path) -> DemoCuration:
    source = Path(path).resolve()
    payload: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("demo curation must be a YAML object")
    return DemoCuration.model_validate(payload)
