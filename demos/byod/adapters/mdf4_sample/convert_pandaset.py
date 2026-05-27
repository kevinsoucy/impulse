#!/usr/bin/env python3
"""Convert a PandaSet recording into a paired MDF4 + sidecar layout.

PandaSet ships as a directory tree of CSV / JPEG / pickle-gzip files. This
script reproduces the equivalent recording in MDF4 form: one `.mf4` file per
PandaSet sequence containing:

  * Scalar bus signals (Vehicle_Speed_kph, Heading_deg, Latitude_deg,
    Longitude_deg) decoded from `meta/gps.csv`, plus derived signals
    (Accel_Longitudinal_ms2, Yaw_Rate_rads) computed from frame-to-frame
    differences of the GPS samples.
  * Per-frame timestamps for the camera, LiDAR, and annotation streams at the
    PandaSet frame rate (10 Hz).
  * String-typed path channels — one per camera, plus one for LiDAR and one
    for annotations — whose values are *relative* paths into a paired sidecar
    directory that holds the unchanged PandaSet binary files.

Real binary data (JPEGs, LiDAR `.pkl.gz`, annotation `.pkl.gz`) is NOT embedded
in MDF4. It stays in a paired directory next to the `.mf4` file. This mirrors
how production OEM recordings are structured.

INPUT layout (PandaSet, unchanged):

    {pandaset_root}/
      {sequence_id}/
        meta/gps.csv
        camera/{cam}/{frame:02d}.jpg
        lidar/{frame:02d}.pkl.gz
        annotations/cuboids/{frame:02d}.pkl.gz
        ...

OUTPUT layout:

    {output_root}/
      pandaset_mdf4/
        {sequence_id}.mf4
      pandaset_mdf4_paired/
        {sequence_id}/
          camera/{cam}/{frame:02d}.jpg            # symlink or copy
          lidar/{frame:02d}.pkl.gz                # symlink or copy
          annotations/cuboids/{frame:02d}.pkl.gz  # symlink or copy

Path channels in the MDF4 carry RELATIVE paths into `pandaset_mdf4_paired/`,
e.g. `{sequence_id}/camera/front_camera/00.jpg`. The reader resolves them
against the paired-directory root at load time.

Usage:

    python convert_pandaset.py --input /path/to/pandaset --output ./out
    python convert_pandaset.py --input /path/to/pandaset --output ./out \\
        --sequence 001
    python convert_pandaset.py --input /path/to/pandaset --list

Attribution: derivative MDF4 carries CC BY 4.0 attribution to Hesai Technology
and Scale AI (PandaSet authors).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# PandaSet's camera names — these match the directory layout under
# {sequence_id}/camera/. The adapter MDF4 emits one string-typed path channel
# per camera.
PANDASET_CAMERAS: tuple[str, ...] = (
    "front_camera",
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_camera",
)

# Frame rate for camera / LiDAR / annotation streams. Real PandaSet GPS may
# differ slightly; scalars use the actual GPS timestamps from meta/gps.csv.
PANDASET_FRAME_DT_S: float = 0.1  # 10 Hz

# m/s → km/h conversion for the speed column in meta/gps.csv.
KPH_PER_MPS: float = 3.6

# Column name → canonical channel name mapping for meta/gps.csv. The PandaSet
# CSVs are not perfectly consistent across releases; common variants covered.
GPS_COLUMN_TO_CHANNEL: dict[str, str] = {
    "speed": "Vehicle_Speed_kph",
    "velocity": "Vehicle_Speed_kph",
    "heading": "Heading_deg",
    "lat": "Latitude_deg",
    "latitude": "Latitude_deg",
    "lon": "Longitude_deg",
    "lng": "Longitude_deg",
    "longitude": "Longitude_deg",
}

# Scale factors applied when copying the CSV value to the canonical channel.
GPS_UNIT_SCALES: dict[str, float] = {
    "Vehicle_Speed_kph": KPH_PER_MPS,  # CSV stores m/s
}


@dataclass(frozen=True)
class ScalarSeries:
    """A single scalar channel as parallel timestamp / value arrays."""

    name: str
    timestamps_s: np.ndarray  # seconds (float64), relative to file start
    values: np.ndarray  # float64


@dataclass(frozen=True)
class PathSeries:
    """A per-frame string-typed channel of relative paths into the paired dir."""

    name: str
    timestamps_s: np.ndarray
    paths: list[str]


@dataclass(frozen=True)
class SequenceData:
    """Everything one PandaSet sequence contributes to one MDF4 file."""

    sequence_id: str
    scalars: list[ScalarSeries]
    path_channels: list[PathSeries]


# ────────────────────────────────────────────────────────────────────────────
# PandaSet input parsing
# ────────────────────────────────────────────────────────────────────────────


def list_sequences(pandaset_root: Path) -> list[str]:
    """Return every directory under pandaset_root that looks like a sequence.

    PandaSet sequence IDs are 3-digit strings (`001`, `002`, …). Any directory
    matching this convention with a `meta/gps.csv` inside is treated as a
    sequence; everything else is ignored.
    """
    if not pandaset_root.is_dir():
        return []
    sequences = []
    for child in sorted(pandaset_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "meta" / "gps.csv").is_file():
            continue
        sequences.append(child.name)
    return sequences


def _read_gps_csv(gps_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Parse meta/gps.csv → {channel_name: [(ts_s, value), ...]}.

    Returns timestamps in seconds (float). Values are scaled to the canonical
    unit declared in GPS_UNIT_SCALES.
    """
    result: dict[str, list[tuple[float, float]]] = {}
    if not gps_path.is_file():
        return result
    with open(gps_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_raw = row.get("timestamp") or row.get("time")
            if ts_raw is None:
                continue
            try:
                ts_s = float(ts_raw)
            except ValueError:
                continue
            for col, value in row.items():
                if col is None or value is None or value == "":
                    continue
                channel = GPS_COLUMN_TO_CHANNEL.get(col.lower())
                if not channel:
                    continue
                try:
                    v = float(value)
                except ValueError:
                    continue
                scale = GPS_UNIT_SCALES.get(channel, 1.0)
                result.setdefault(channel, []).append((ts_s, v * scale))
    return result


def _derived_from_speed(speed_series: ScalarSeries) -> ScalarSeries:
    """Compute longitudinal acceleration (m/s²) from Vehicle_Speed_kph deltas.

    First sample carries the same value as the second to avoid a NaN at index 0;
    this is the same edge-handling the existing PandaSet adapter uses.
    """
    if speed_series.values.size < 2:
        return ScalarSeries(
            name="Accel_Longitudinal_ms2",
            timestamps_s=speed_series.timestamps_s.copy(),
            values=np.zeros_like(speed_series.values),
        )
    speed_ms = speed_series.values / KPH_PER_MPS
    dt = np.diff(speed_series.timestamps_s)
    dt = np.where(dt <= 0, np.median(dt), dt)
    accel = np.empty_like(speed_ms)
    accel[1:] = np.diff(speed_ms) / dt
    accel[0] = accel[1]
    return ScalarSeries(
        name="Accel_Longitudinal_ms2",
        timestamps_s=speed_series.timestamps_s.copy(),
        values=accel,
    )


def _derived_from_heading(heading_series: ScalarSeries) -> ScalarSeries:
    """Yaw rate (rad/s) from Heading_deg deltas, wrap-corrected."""
    if heading_series.values.size < 2:
        return ScalarSeries(
            name="Yaw_Rate_rads",
            timestamps_s=heading_series.timestamps_s.copy(),
            values=np.zeros_like(heading_series.values),
        )
    heading_rad = np.unwrap(np.deg2rad(heading_series.values))
    dt = np.diff(heading_series.timestamps_s)
    dt = np.where(dt <= 0, np.median(dt), dt)
    yaw_rate = np.empty_like(heading_rad)
    yaw_rate[1:] = np.diff(heading_rad) / dt
    yaw_rate[0] = yaw_rate[1]
    return ScalarSeries(
        name="Yaw_Rate_rads",
        timestamps_s=heading_series.timestamps_s.copy(),
        values=yaw_rate,
    )


def _to_scalar_series(name: str, samples: list[tuple[float, float]]) -> ScalarSeries:
    """Sort by timestamp and split into two parallel arrays.

    The CSV is expected to be in time order but defensiveness costs nothing.
    """
    samples_sorted = sorted(samples, key=lambda x: x[0])
    ts = np.array([s[0] for s in samples_sorted], dtype=np.float64)
    vs = np.array([s[1] for s in samples_sorted], dtype=np.float64)
    return ScalarSeries(name=name, timestamps_s=ts - ts[0] if ts.size else ts, values=vs)


def _enumerate_frame_paths(sequence_dir: Path, sequence_id: str) -> list[PathSeries]:
    """Discover per-frame camera / LiDAR / annotation files and emit path channels.

    Returns one PathSeries per camera, one for LiDAR, one for annotations. All
    paths are RELATIVE to the paired-directory root and prefixed with the
    sequence_id so they resolve unambiguously when many sequences share one
    paired root.
    """
    lidar_dir = sequence_dir / "lidar"
    if not lidar_dir.is_dir():
        return []
    # Frame indices come from the LiDAR directory; camera/annotation files
    # follow the same `{frame:02d}` convention. Sorted for determinism.
    # NOTE: Path.stem only strips one extension, so `00.pkl.gz`.stem == `00.pkl`;
    # we strip the full `.pkl.gz` suffix explicitly.
    frame_indices: list[int] = []
    for pkl in sorted(lidar_dir.glob("*.pkl.gz")):
        name = pkl.name
        if name.endswith(".pkl.gz"):
            name = name[: -len(".pkl.gz")]
        try:
            frame_indices.append(int(name))
        except ValueError:
            continue
    if not frame_indices:
        return []
    timestamps_s = np.array(
        [i * PANDASET_FRAME_DT_S for i in range(len(frame_indices))],
        dtype=np.float64,
    )
    out: list[PathSeries] = []
    for cam in PANDASET_CAMERAS:
        if not (sequence_dir / "camera" / cam).is_dir():
            continue
        paths = [f"{sequence_id}/camera/{cam}/{fi:02d}.jpg" for fi in frame_indices]
        out.append(PathSeries(name=f"{cam.upper()}_path", timestamps_s=timestamps_s, paths=paths))
    out.append(
        PathSeries(
            name="LIDAR_path",
            timestamps_s=timestamps_s,
            paths=[f"{sequence_id}/lidar/{fi:02d}.pkl.gz" for fi in frame_indices],
        )
    )
    out.append(
        PathSeries(
            name="ANNOTATIONS_path",
            timestamps_s=timestamps_s,
            paths=[f"{sequence_id}/annotations/cuboids/{fi:02d}.pkl.gz" for fi in frame_indices],
        )
    )
    return out


def parse_sequence(pandaset_root: Path, sequence_id: str) -> SequenceData:
    """Read everything we need for one sequence and shape it into SequenceData."""
    sequence_dir = pandaset_root / sequence_id
    gps_raw = _read_gps_csv(sequence_dir / "meta" / "gps.csv")
    scalars: list[ScalarSeries] = []
    for channel, samples in gps_raw.items():
        if samples:
            scalars.append(_to_scalar_series(channel, samples))
    speed = next((s for s in scalars if s.name == "Vehicle_Speed_kph"), None)
    heading = next((s for s in scalars if s.name == "Heading_deg"), None)
    if speed is not None:
        scalars.append(_derived_from_speed(speed))
    if heading is not None:
        scalars.append(_derived_from_heading(heading))
    path_channels = _enumerate_frame_paths(sequence_dir, sequence_id)
    return SequenceData(
        sequence_id=sequence_id,
        scalars=scalars,
        path_channels=path_channels,
    )


# ────────────────────────────────────────────────────────────────────────────
# MDF4 writer
# ────────────────────────────────────────────────────────────────────────────


def _build_signal_groups(sequence: SequenceData) -> list[list]:
    """Construct asammdf Signal objects grouped by sample rate / master timestamp.

    Each inner list becomes one MDF4 channel group with its own master channel.
    The conversion always produces at most two groups: scalars (at GPS rate) and
    path channels (at the PandaSet 10 Hz frame rate). Mixing rates inside a
    single group would force asammdf to interpolate, which is not what we want.

    Imported lazily so the module stays importable on systems without asammdf.
    """
    from asammdf import Signal  # noqa: PLC0415

    groups: list[list] = []

    scalar_signals: list = []
    for s in sequence.scalars:
        scalar_signals.append(
            Signal(
                samples=s.values.astype(np.float64),
                timestamps=s.timestamps_s.astype(np.float64),
                name=s.name,
            )
        )
    if scalar_signals:
        groups.append(scalar_signals)

    path_signals: list = []
    for p in sequence.path_channels:
        # MDF4 string channels are byte-typed (data_type STRING_LATIN_1 / UTF-8);
        # asammdf 8.x rejects numpy unicode (`U`) dtypes and requires `S{N}` with
        # an explicit `encoding` set on the Signal.
        encoded = [x.encode("utf-8") for x in p.paths]
        max_len = max((len(x) for x in encoded), default=1)
        samples = np.array(encoded, dtype=f"S{max_len}")
        path_signals.append(
            Signal(
                samples=samples,
                timestamps=p.timestamps_s.astype(np.float64),
                name=p.name,
                encoding="utf-8",
            )
        )
    if path_signals:
        groups.append(path_signals)

    return groups


def write_mdf4(sequence: SequenceData, out_path: Path) -> None:
    """Write one MDF4 file for one PandaSet sequence."""
    from asammdf import MDF  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mdf = MDF(version="4.10")
    for group in _build_signal_groups(sequence):
        mdf.append(group)
    mdf.save(out_path, overwrite=True)
    mdf.close()


def stage_paired_directory(
    pandaset_root: Path,
    sequence_id: str,
    paired_root: Path,
    *,
    copy: bool = False,
) -> None:
    """Materialise the paired-directory layout for one sequence.

    Uses symlinks by default (fast, zero-byte) and copies only when explicitly
    requested. The destination tree mirrors PandaSet's original layout under
    `paired_root/{sequence_id}/`.
    """
    src = pandaset_root / sequence_id
    dst = paired_root / sequence_id
    for sub in ("camera", "lidar", "annotations"):
        src_sub = src / sub
        if not src_sub.is_dir():
            continue
        for path in src_sub.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            if copy:
                shutil.copy2(path, target)
            else:
                target.symlink_to(path.resolve())


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", required=True, type=Path, help="Root of an extracted PandaSet dataset."
    )
    p.add_argument("--output", type=Path, help="Root for the MDF4 + paired-directory output.")
    p.add_argument(
        "--sequence",
        default=None,
        help="Convert only this sequence ID. Default: all sequences in --input.",
    )
    p.add_argument(
        "--list", action="store_true", help="List discovered sequence IDs and exit (no writes)."
    )
    p.add_argument(
        "--copy", action="store_true", help="Copy paired files instead of symlinking them."
    )
    p.add_argument(
        "--no-stage-paired",
        action="store_true",
        help="Skip materialising the paired directory (write MDF4 only).",
    )
    args = p.parse_args(argv)

    pandaset_root = args.input.expanduser().resolve()
    sequences = list_sequences(pandaset_root)
    if not sequences:
        print(f"No PandaSet sequences found under {pandaset_root}.", file=sys.stderr)
        return 1
    if args.list:
        for seq in sequences:
            print(seq)
        return 0
    if args.sequence:
        if args.sequence not in sequences:
            print(f"Sequence {args.sequence!r} not found in {pandaset_root}.", file=sys.stderr)
            return 1
        targets: Iterable[str] = [args.sequence]
    else:
        targets = sequences

    if args.output is None:
        print("--output is required unless --list is set.", file=sys.stderr)
        return 2
    output_root = args.output.expanduser().resolve()
    mdf4_root = output_root / "pandaset_mdf4"
    paired_root = output_root / "pandaset_mdf4_paired"

    for sequence_id in targets:
        data = parse_sequence(pandaset_root, sequence_id)
        out_mf4 = mdf4_root / f"{sequence_id}.mf4"
        write_mdf4(data, out_mf4)
        if not args.no_stage_paired:
            stage_paired_directory(pandaset_root, sequence_id, paired_root, copy=args.copy)
        print(f"✓ {sequence_id} → {out_mf4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
