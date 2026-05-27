"""Round-trip tests for the MDF4 channel patterns the PandaSet → MDF4
conversion script relies on.

Each test writes a small MDF4 with asammdf, reads it back with asammdf, and
asserts the values survived the round trip. The test does NOT depend on the
PandaSet dataset or on Impulse's centralized reader — it asserts that the
MDF4 file is structurally correct on its own terms, which is a property of
the conversion patterns plus asammdf.

Four layers:

1. Path channels (string-typed) — paths written into a fixed-width unicode
   channel come back byte-identical.
2. Scalar signals (float) — numeric channels round-trip within 1e-6 tolerance.
3. Timestamp alignment — input and output timestamps match within 1 µs.
4. End-to-end resolve — paths read back from MDF4 resolve against a paired
   directory of dummy files, which open without error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from asammdf import MDF, Signal

# ── Helpers ────────────────────────────────────────────────────────────────


def _write_mdf4(signals: list[Signal], path: Path) -> None:
    """Write one channel group containing all signals.

    All signals must share the same timestamps; mixing sample rates in a single
    group makes asammdf interpolate, which is not what these tests check.
    Use `_write_mdf4_groups` for multi-rate cases.
    """
    mdf = MDF(version="4.10")
    mdf.append(signals)
    mdf.save(path, overwrite=True)
    mdf.close()


def _write_mdf4_groups(groups: list[list[Signal]], path: Path) -> None:
    """Write multiple channel groups, one per `signals` list passed in.

    Use this when signals belong to different sample rates — each rate gets
    its own master (timestamp) channel by living in its own channel group.
    """
    mdf = MDF(version="4.10")
    for signals in groups:
        mdf.append(signals)
    mdf.save(path, overwrite=True)
    mdf.close()


def _read_channel(path: Path, name: str):
    with MDF(path) as mdf:
        sig = mdf.get(name)
    return sig


def _decode_string_samples(samples: np.ndarray) -> list[str]:
    """asammdf returns string-channel samples as a numpy array of bytes
    (MDF4 strings are byte-typed). Decode each entry to a Python str and
    strip the trailing nulls that fixed-width byte arrays carry on shorter
    entries."""
    out: list[str] = []
    for s in samples:
        if isinstance(s, bytes):
            out.append(s.rstrip(b"\x00").decode("utf-8"))
        else:
            out.append(str(s).rstrip("\x00"))
    return out


def _encode_paths_as_bytes_array(paths: list[str]) -> np.ndarray:
    """Encode a list of paths as a fixed-width bytes array suitable for
    asammdf's MDF4 string-channel writer."""
    encoded = [p.encode("utf-8") for p in paths]
    max_len = max(len(p) for p in encoded)
    return np.array(encoded, dtype=f"S{max_len}")


# ── 1. Path channels ───────────────────────────────────────────────────────


def test_path_channel_round_trips_byte_identical(tmp_path: Path):
    """Write a list of PandaSet-shaped relative paths into a string-typed
    channel, read back, and assert every entry is byte-identical to input."""
    paths = [
        "001/camera/front_camera/00.jpg",
        "001/camera/front_camera/01.jpg",
        "001/camera/front_camera/02.jpg",
        "001/camera/front_left_camera/00.jpg",
        "001/camera/front_left_camera/01.jpg",
        "001/camera/front_left_camera/02.jpg",
    ]
    timestamps_s = np.arange(len(paths), dtype=np.float64) * 0.1  # 10 Hz
    samples = _encode_paths_as_bytes_array(paths)
    sig = Signal(samples=samples, timestamps=timestamps_s, name="CAMERA_path", encoding="utf-8")

    out = tmp_path / "paths.mf4"
    _write_mdf4([sig], out)

    recovered = _read_channel(out, "CAMERA_path")
    decoded = _decode_string_samples(recovered.samples)

    assert decoded == paths, (
        f"Path channel did not round-trip byte-identical.\n"
        f"  input  = {paths}\n"
        f"  output = {decoded}"
    )


def test_path_channel_handles_variable_length_paths(tmp_path: Path):
    """PandaSet's annotation paths are longer than camera paths. Mixing path
    lengths in a single channel is the common case; ensure short paths don't
    get truncated or padded with nulls."""
    paths = [
        "001/lidar/00.pkl.gz",
        "001/annotations/cuboids/00.pkl.gz",  # longer
        "001/lidar/01.pkl.gz",
    ]
    timestamps_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    samples = _encode_paths_as_bytes_array(paths)
    sig = Signal(samples=samples, timestamps=timestamps_s, name="MIXED_path", encoding="utf-8")

    out = tmp_path / "mixed_paths.mf4"
    _write_mdf4([sig], out)

    decoded = _decode_string_samples(_read_channel(out, "MIXED_path").samples)
    assert decoded == paths


# ── 2. Scalar signals (float) ──────────────────────────────────────────────


def test_scalar_signals_round_trip_within_float_tolerance(tmp_path: Path):
    """Three representative PandaSet-derived scalars at 10 Hz round-trip to
    within 1e-6."""
    n = 100
    timestamps_s = np.arange(n, dtype=np.float64) * 0.1  # 10 Hz
    speed_kph = 30.0 + 5.0 * np.sin(np.arange(n) * 0.1)
    steering_deg = -2.0 + 0.5 * np.cos(np.arange(n) * 0.1)
    accel_ms2 = np.gradient(speed_kph / 3.6, 0.1)

    signals = [
        Signal(samples=speed_kph, timestamps=timestamps_s, name="Vehicle_Speed_kph"),
        Signal(samples=steering_deg, timestamps=timestamps_s, name="Steering_Angle_deg"),
        Signal(samples=accel_ms2, timestamps=timestamps_s, name="Accel_Longitudinal_ms2"),
    ]

    out = tmp_path / "scalars.mf4"
    _write_mdf4(signals, out)

    for sig in signals:
        recovered = _read_channel(out, sig.name)
        assert np.allclose(recovered.samples, sig.samples, atol=1e-6), (
            f"Scalar channel {sig.name!r} did not round-trip within 1e-6.\n"
            f"  max abs diff = {np.max(np.abs(recovered.samples - sig.samples))}"
        )


# ── 3. Timestamp alignment ─────────────────────────────────────────────────


def test_timestamps_round_trip_within_one_microsecond(tmp_path: Path):
    """Timestamps at the 10 Hz perception rate and the 100 Hz scalar rate in
    the same file both come back within 1 µs of input."""
    perception_ts = np.arange(50, dtype=np.float64) * 0.1  # 10 Hz, 5 s
    scalar_ts = np.arange(500, dtype=np.float64) * 0.01  # 100 Hz, 5 s

    paths = [f"001/camera/front_camera/{i:02d}.jpg" for i in range(50)]
    perception_sig = Signal(
        samples=_encode_paths_as_bytes_array(paths),
        timestamps=perception_ts,
        name="FRONT_CAMERA_path",
        encoding="utf-8",
    )
    scalar_sig = Signal(
        samples=np.ones(500, dtype=np.float64) * 30.0,
        timestamps=scalar_ts,
        name="Vehicle_Speed_kph",
    )

    out = tmp_path / "timestamps.mf4"
    _write_mdf4_groups([[perception_sig], [scalar_sig]], out)

    perception_back = _read_channel(out, "FRONT_CAMERA_path")
    scalar_back = _read_channel(out, "Vehicle_Speed_kph")

    one_us_s = 1e-6
    assert (
        np.max(np.abs(perception_back.timestamps - perception_ts)) < one_us_s
    ), f"Perception timestamps drifted by {np.max(np.abs(perception_back.timestamps - perception_ts)) * 1e6:.3f} µs"
    assert (
        np.max(np.abs(scalar_back.timestamps - scalar_ts)) < one_us_s
    ), f"Scalar timestamps drifted by {np.max(np.abs(scalar_back.timestamps - scalar_ts)) * 1e6:.3f} µs"


# ── 4. End-to-end resolve ──────────────────────────────────────────────────


def test_paths_read_back_resolve_against_paired_directory(tmp_path: Path):
    """Write paths into MDF4 that point at dummy files in a sibling paired
    directory; read paths back; resolve each one and open the referenced file.
    This is the integration check that the whole chain wires up correctly."""
    paired_root = tmp_path / "paired"
    sequence_id = "001"

    # Build a one-frame paired directory with placeholder bytes in each slot.
    relpaths = [
        f"{sequence_id}/camera/front_camera/00.jpg",
        f"{sequence_id}/lidar/00.pkl.gz",
        f"{sequence_id}/annotations/cuboids/00.pkl.gz",
    ]
    sentinels = {
        relpaths[0]: b"JPEG_PLACEHOLDER",
        relpaths[1]: b"LIDAR_PLACEHOLDER",
        relpaths[2]: b"ANNOTATION_PLACEHOLDER",
    }
    for rel, payload in sentinels.items():
        full = paired_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(payload)

    timestamps_s = np.array([0.0], dtype=np.float64)
    signals = []
    for rel in relpaths:
        name = rel.split("/")[1].upper() + "_path" if "camera/" not in rel else "FRONT_CAMERA_path"
        signals.append(
            Signal(
                samples=_encode_paths_as_bytes_array([rel]),
                timestamps=timestamps_s,
                name=name,
                encoding="utf-8",
            )
        )

    out = tmp_path / "resolve.mf4"
    _write_mdf4(signals, out)

    # Read each channel back, resolve the path, open the file, assert payload
    # matches the sentinel we wrote.
    for sig in signals:
        recovered = _read_channel(out, sig.name)
        decoded = _decode_string_samples(recovered.samples)
        assert len(decoded) == 1
        resolved = paired_root / decoded[0]
        assert resolved.is_file(), f"Resolved path does not exist: {resolved}"
        assert resolved.read_bytes() == sentinels[decoded[0]], f"Payload mismatch for {decoded[0]}"


# ── 5. Conversion-script smoke test ────────────────────────────────────────


def test_convert_pandaset_runs_against_synthetic_input(tmp_path: Path):
    """The conversion script's CLI smoke test: build a one-sequence synthetic
    PandaSet tree, run the conversion, and check that the produced MDF4 has
    the expected channels."""
    # Import here so the asammdf-only tests above run even if the conversion
    # script has unrelated import issues.
    from adapters.mdf4_sample import convert_pandaset

    pandaset_root = tmp_path / "pandaset"
    sequence_id = "001"
    sequence_dir = pandaset_root / sequence_id
    (sequence_dir / "meta").mkdir(parents=True)
    (sequence_dir / "camera" / "front_camera").mkdir(parents=True)
    (sequence_dir / "lidar").mkdir(parents=True)
    (sequence_dir / "annotations" / "cuboids").mkdir(parents=True)

    # Minimal gps.csv at 10 Hz across 0.5 s.
    gps = sequence_dir / "meta" / "gps.csv"
    gps.write_text(
        "timestamp,speed,heading,lat,lon\n"
        + "\n".join(
            f"{i * 0.1:.2f},{8.0 + i * 0.1},{45.0 + i},{37.0 + i * 1e-6},{-122.0 + i * 1e-6}"
            for i in range(5)
        )
        + "\n"
    )
    # Five frames in each modality.
    for i in range(5):
        (sequence_dir / "camera" / "front_camera" / f"{i:02d}.jpg").write_bytes(b"j")
        (sequence_dir / "lidar" / f"{i:02d}.pkl.gz").write_bytes(b"l")
        (sequence_dir / "annotations" / "cuboids" / f"{i:02d}.pkl.gz").write_bytes(b"a")

    output_root = tmp_path / "out"
    rc = convert_pandaset.main(
        [
            "--input",
            str(pandaset_root),
            "--output",
            str(output_root),
            "--no-stage-paired",
        ]
    )
    assert rc == 0

    mdf_path = output_root / "pandaset_mdf4" / f"{sequence_id}.mf4"
    assert mdf_path.is_file()

    # Spot-check that the expected channels exist.
    with MDF(mdf_path) as mdf:
        channel_names = {ch.name for group in mdf.groups for ch in group.channels}
    expected = {
        "Vehicle_Speed_kph",
        "Heading_deg",
        "Latitude_deg",
        "Longitude_deg",
        "Accel_Longitudinal_ms2",
        "Yaw_Rate_rads",
        "FRONT_CAMERA_path",
        "LIDAR_path",
        "ANNOTATIONS_path",
    }
    missing = expected - channel_names
    assert not missing, f"MDF4 is missing expected channels: {missing}"
