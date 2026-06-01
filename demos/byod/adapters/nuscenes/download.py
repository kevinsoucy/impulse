"""Download / validate the NuScenes dataset on disk.

NuScenes terms of use require registration and EULA acceptance. We do NOT bake
URLs into this module — the caller obtains a signed URL from
https://www.nuscenes.org/download and either passes it to `download(url=...)`
or pre-extracts the data into the expected dataroot layout.
"""

from __future__ import annotations

import tarfile
from pathlib import Path


# NuScenes ships these subdirectories under dataroot. Used to validate that an
# existing dataroot is a complete extraction (not a partial download). `sweeps/`
# (the un-annotated non-keyframe sensor frames) is intentionally NOT required: the
# demo stages only keyframe `samples/`, and perception_paths indexes keyframes
# only. Keep `samples/` + the per-version metadata dir as the hard requirement.
EXPECTED_SUBDIRS = ("maps", "samples")


def has_complete_extraction(dataroot: Path, version: str) -> bool:
    if not dataroot.is_dir():
        return False
    for sub in EXPECTED_SUBDIRS:
        if not (dataroot / sub).is_dir():
            return False
    if not (dataroot / version).is_dir():
        return False
    return True


def validate_dataroot(dataroot: Path, version: str) -> None:
    """Raise FileNotFoundError if the dataroot is missing or partial."""
    if not has_complete_extraction(dataroot, version):
        raise FileNotFoundError(
            f"NuScenes dataroot {dataroot} is missing or incomplete for {version!r}. "
            f"Expected subdirectories: {', '.join((*EXPECTED_SUBDIRS, version))}. "
            f"Download from https://www.nuscenes.org/download (EULA required)."
        )


def download_to(url: str, target: Path) -> None:
    """Stream-download a URL to `target`. Uses urllib so no extra deps required."""
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(target, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 1024
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB ({pct:.1f}%)", end="")
        print()


def extract_tarball(tarball: Path, dataroot: Path) -> None:
    dataroot.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dataroot, filter="data")  # filter='data' guards against path traversal
