"""Shared test setup: repo root on sys.path and the two configuration files."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def default_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def smoke_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
