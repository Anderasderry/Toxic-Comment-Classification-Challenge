"""Shared run timestamp for submission files and figure directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def new_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def path_with_timestamp(path: Path, run_ts: str) -> Path:
    return path.with_name(f"{path.stem}_{run_ts}{path.suffix}")


def fig_run_dir(figs_root: Path, model_slug: str, run_ts: str) -> Path:
    out = figs_root.expanduser().resolve() / f"{model_slug}_{run_ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out
