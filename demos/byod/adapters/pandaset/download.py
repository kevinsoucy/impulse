"""Validate that a PandaSet dataroot is extracted.

PandaSet is distributed via the Hesai data portal (registration required for
the v1 release after Scale AI's hosting moved to Hesai in 2022). This module
validates layout — it doesn't fetch.
"""

from __future__ import annotations

import re
from pathlib import Path


_SEQUENCE_DIR_RE = re.compile(r"^\d{3}$")


def has_complete_extraction(dataroot: Path) -> bool:
    if not dataroot.is_dir():
        return False
    for entry in dataroot.iterdir():
        if entry.is_dir() and _SEQUENCE_DIR_RE.match(entry.name):
            return True
    return False


def validate_dataroot(dataroot: Path, version: str = "v1") -> None:
    if not has_complete_extraction(dataroot):
        raise FileNotFoundError(
            f"PandaSet dataroot {dataroot} is missing or incomplete for {version!r}. "
            f"Expected: at least one 3-digit sequence directory (e.g. {dataroot}/001/). "
            f"Download from https://pandaset.org/ via the Hesai data portal "
            f"(registration required, CC BY 4.0)."
        )
