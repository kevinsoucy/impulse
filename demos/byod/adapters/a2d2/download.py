"""Validate that an A2D2 dataroot is fully extracted.

A2D2 requires registering at https://www.a2d2.audi/a2d2/en/download.html
to obtain the download URLs (no EULA acceptance flow in code). The bundles
target an extracted dataroot, so this module only validates layout rather
than driving the download itself.
"""

from __future__ import annotations

from pathlib import Path


# Subdirectories the loader walks. Missing any of them means the dataroot is
# either incomplete or pointed at the wrong path.
def expected_subdirs(version: str) -> tuple[str, ...]:
    """Returns the subdirectories the A2D2 loader requires under `{dataroot}/`."""
    return ("cams_lidars.json", version)


def has_complete_extraction(dataroot: Path, version: str) -> bool:
    if not dataroot.is_dir():
        return False
    if not (dataroot / "cams_lidars.json").is_file():
        return False
    if not (dataroot / version).is_dir():
        return False
    # At least one scene directory.
    for entry in (dataroot / version).iterdir():
        if entry.is_dir():
            return True
    return False


def validate_dataroot(dataroot: Path, version: str) -> None:
    """Raise FileNotFoundError if the dataroot is missing or partial."""
    if not has_complete_extraction(dataroot, version):
        raise FileNotFoundError(
            f"A2D2 dataroot {dataroot} is missing or incomplete for {version!r}. "
            f"Expected: {dataroot}/cams_lidars.json and at least one scene under "
            f"{dataroot}/{version}/. Download from "
            f"https://www.a2d2.audi/a2d2/en/download.html (registration required, "
            f"CC BY-ND 4.0)."
        )
