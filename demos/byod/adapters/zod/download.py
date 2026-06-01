"""Validate that a ZOD dataroot is fully extracted.

ZOD requires registering at https://zod.zenseact.com/download/ for download
URLs. This module validates layout — it doesn't drive the fetch.
"""

from __future__ import annotations

from pathlib import Path


def has_complete_extraction(dataroot: Path, version: str) -> bool:
    if not dataroot.is_dir():
        return False
    # Two valid layouts:
    #   {dataroot}/sequences/{seq_id}/...                  (direct)
    #   {dataroot}/{version}/sequences/{seq_id}/...        (version-prefixed)
    candidates = [dataroot / "sequences", dataroot / version / "sequences"]
    for cand in candidates:
        if cand.is_dir():
            for entry in cand.iterdir():
                if entry.is_dir():
                    return True
    return False


def validate_dataroot(dataroot: Path, version: str) -> None:
    if not has_complete_extraction(dataroot, version):
        raise FileNotFoundError(
            f"ZOD dataroot {dataroot} is missing or incomplete for {version!r}. "
            f"Expected: at least one sequence under "
            f"{dataroot}/sequences/ or {dataroot}/{version}/sequences/. "
            f"Download from https://zod.zenseact.com/download/ "
            f"(registration required, CC BY-SA 4.0)."
        )
