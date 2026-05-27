"""Unit tests for demos/byod/lib/visualization.py.

The reader functions are pure-Python and decode synthetic binary fixtures
written to ``tmp_path``. Channel-resolution helpers require Spark and are
covered in the integration tier (omitted here)."""

from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_ROOT = Path(__file__).resolve().parents[3] / "demos" / "byod"
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from lib.visualization import (  # noqa: E402
    adapter_visualize_format,
    read_camera_image,
    read_lidar_xyzi,
)


# ── read_lidar_xyzi ─────────────────────────────────────────────────────────


class TestReadLidarXyzi:
    def test_fromfile_float32_stride5(self, tmp_path: Path):
        # 4 points × (x, y, z, intensity, ring) = 20 float32 values.
        points = np.array([
            [1.0, 2.0, 3.0, 0.5, 0],
            [4.0, 5.0, 6.0, 0.6, 1],
            [-1.0, -2.0, -3.0, 0.7, 2],
            [0.0, 0.0, 0.0, 0.8, 3],
        ], dtype=np.float32)
        f = tmp_path / "scan.bin"
        points.tofile(f)

        x, y, z, intensity = read_lidar_xyzi(str(f), reader="fromfile", dtype="float32", stride=5)
        np.testing.assert_allclose(x, [1.0, 4.0, -1.0, 0.0])
        np.testing.assert_allclose(y, [2.0, 5.0, -2.0, 0.0])
        np.testing.assert_allclose(z, [3.0, 6.0, -3.0, 0.0])
        np.testing.assert_allclose(intensity, [0.5, 0.6, 0.7, 0.8])

    def test_npz_first_array(self, tmp_path: Path):
        arr = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.6]], dtype=np.float32)
        f = tmp_path / "scan.npz"
        np.savez(f, arr)
        x, y, z, intensity = read_lidar_xyzi(str(f), reader="npz")
        np.testing.assert_allclose(x, [1.0, 4.0])
        np.testing.assert_allclose(intensity, [0.5, 0.6])

    def test_pickle_gz(self, tmp_path: Path):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        df = pd.DataFrame({
            "x": [1.0, 4.0],
            "y": [2.0, 5.0],
            "z": [3.0, 6.0],
            "i": [0.5, 0.6],
            "d": [0, 1],
        })
        f = tmp_path / "scan.pkl.gz"
        with gzip.open(f, "wb") as gz:
            pickle.dump(df, gz)
        x, y, z, intensity = read_lidar_xyzi(str(f), reader="pickle_gz")
        np.testing.assert_allclose(x, [1.0, 4.0])
        np.testing.assert_allclose(intensity, [0.5, 0.6])

    def test_unknown_reader_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unknown lidar_reader"):
            read_lidar_xyzi(str(tmp_path / "x"), reader="not_a_reader")


# ── read_camera_image ───────────────────────────────────────────────────────


class TestReadCameraImage:
    def test_pil_reader(self, tmp_path: Path):
        pil = pytest.importorskip("PIL.Image")
        # Write a tiny PNG via PIL itself so the round-trip is deterministic.
        img = pil.new("RGB", (4, 3), color=(10, 20, 30))
        f = tmp_path / "frame.png"
        img.save(f)

        arr = read_camera_image(str(f), reader="pil")
        assert arr.shape == (3, 4, 3)
        # All pixels are (10, 20, 30).
        np.testing.assert_array_equal(arr[0, 0], [10, 20, 30])

    def test_unknown_reader_raises(self):
        with pytest.raises(ValueError, match="unknown camera_reader"):
            read_camera_image("/nonexistent", reader="not_a_reader")


# ── adapter_visualize_format ────────────────────────────────────────────────


class TestAdapterVisualizeFormat:
    def test_defaults_when_adapter_returns_empty(self):
        class FakeAdapter:
            def visualize_format(self):
                return {}

        fmt = adapter_visualize_format(FakeAdapter())
        assert fmt == {
            "camera_reader": "pil",
            "lidar_reader": "fromfile",
            "lidar_dtype": "float32",
            "lidar_stride": 5,
        }

    def test_overrides_apply(self):
        class FakeAdapter:
            def visualize_format(self):
                return {"camera_reader": "opencv", "lidar_stride": 4}

        fmt = adapter_visualize_format(FakeAdapter())
        assert fmt["camera_reader"] == "opencv"
        assert fmt["lidar_stride"] == 4
        # Unset keys still get defaults.
        assert fmt["lidar_reader"] == "fromfile"
        assert fmt["lidar_dtype"] == "float32"

    def test_handles_none_return(self):
        class FakeAdapter:
            def visualize_format(self):
                return None

        fmt = adapter_visualize_format(FakeAdapter())
        assert fmt["camera_reader"] == "pil"
