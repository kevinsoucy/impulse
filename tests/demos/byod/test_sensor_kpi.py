"""Unit tests for demos/byod/lib/sensor_kpi.py.

The helpers are adapter-agnostic — they operate on the pipe-delimited
`source` string convention every adapter populates in `object_tracks.source`.
Tests live at the byod root, not under any adapter, to reflect that.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_DEMO_ROOT = Path(__file__).resolve().parents[3] / "demos" / "byod"
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from lib.sensor_kpi import (  # noqa: E402
    class_coverage_by_modality,
    detection_counts_by_source,
    detection_gaps,
    distance_stats_by_modality,
    sensor_flags,
    with_sensor_flags,
)


# ── sensor_flags ─────────────────────────────────────────────────────────────


class TestSensorFlags:
    def test_all_modalities(self):
        f = sensor_flags("lidar|radar|camera")
        assert f == {"lidar": True, "radar": True, "camera": True}

    def test_lidar_only(self):
        f = sensor_flags("lidar")
        assert f == {"lidar": True, "radar": False, "camera": False}

    def test_camera_only(self):
        f = sensor_flags("camera")
        assert f == {"lidar": False, "radar": False, "camera": True}

    def test_lidar_camera(self):
        f = sensor_flags("lidar|camera")
        assert f == {"lidar": True, "radar": False, "camera": True}

    def test_empty_string_all_false(self):
        f = sensor_flags("")
        assert f == {"lidar": False, "radar": False, "camera": False}

    def test_unknown_token_ignored(self):
        f = sensor_flags("unknown_sensor")
        assert f == {"lidar": False, "radar": False, "camera": False}


# ── with_sensor_flags ────────────────────────────────────────────────────────


def _df(rows):
    return pd.DataFrame(rows)


class TestWithSensorFlags:
    def test_adds_three_columns(self):
        df = _df([{"source": "lidar|radar|camera", "container_id": 1, "frame_ts": 0, "distance_m": 5.0}])
        out = with_sensor_flags(df)
        assert "lidar_seen" in out.columns
        assert "radar_seen" in out.columns
        assert "camera_seen" in out.columns

    def test_preserves_original_columns(self):
        df = _df([{"source": "camera", "container_id": 1, "frame_ts": 0, "distance_m": 10.0}])
        out = with_sensor_flags(df)
        assert "source" in out.columns
        assert "container_id" in out.columns

    def test_flag_values_correct_lidar_only(self):
        df = _df([{"source": "lidar", "container_id": 1, "frame_ts": 0, "distance_m": 5.0}])
        out = with_sensor_flags(df)
        row = out.iloc[0]
        assert bool(row["lidar_seen"])
        assert not bool(row["radar_seen"])
        assert not bool(row["camera_seen"])

    def test_flag_values_all_modalities(self):
        df = _df([{"source": "lidar|radar|camera", "container_id": 1, "frame_ts": 0, "distance_m": 5.0}])
        out = with_sensor_flags(df)
        row = out.iloc[0]
        assert bool(row["lidar_seen"])
        assert bool(row["radar_seen"])
        assert bool(row["camera_seen"])

    def test_multiple_rows_flagged_independently(self):
        df = _df([
            {"source": "lidar", "container_id": 1, "frame_ts": 0, "distance_m": 5.0},
            {"source": "camera", "container_id": 1, "frame_ts": 1, "distance_m": 15.0},
        ])
        out = with_sensor_flags(df)
        assert bool(out.iloc[0]["lidar_seen"])
        assert not bool(out.iloc[0]["camera_seen"])
        assert not bool(out.iloc[1]["lidar_seen"])
        assert bool(out.iloc[1]["camera_seen"])


# ── detection_counts_by_source ───────────────────────────────────────────────


class TestDetectionCountsBySource:
    def _make_df(self):
        return _df([
            {"container_id": 1, "source": "lidar|camera"},
            {"container_id": 1, "source": "lidar|camera"},
            {"container_id": 1, "source": "camera"},
            {"container_id": 2, "source": "lidar"},
        ])

    def test_returns_dataframe(self):
        out = detection_counts_by_source(self._make_df())
        assert isinstance(out, pd.DataFrame)

    def test_grouped_counts_correct(self):
        out = detection_counts_by_source(self._make_df())
        row = out[(out["container_id"] == 1) & (out["source"] == "lidar|camera")]
        assert len(row) == 1
        assert int(row["detection_count"].iloc[0]) == 2

    def test_sorted_descending(self):
        out = detection_counts_by_source(self._make_df())
        counts = list(out["detection_count"])
        assert counts == sorted(counts, reverse=True)

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["container_id", "source"])
        out = detection_counts_by_source(df)
        assert len(out) == 0


# ── distance_stats_by_modality ───────────────────────────────────────────────


class TestDistanceStatsByModality:
    def _make_df(self):
        df = _df([
            {"source": "lidar", "distance_m": 10.0, "container_id": 1},
            {"source": "lidar", "distance_m": 20.0, "container_id": 1},
            {"source": "lidar", "distance_m": 30.0, "container_id": 1},
            {"source": "camera", "distance_m": 5.0, "container_id": 1},
            {"source": "radar", "distance_m": 50.0, "container_id": 1},
        ])
        return with_sensor_flags(df)

    def test_returns_dataframe_with_expected_columns(self):
        out = distance_stats_by_modality(self._make_df())
        assert "modality" in out.columns
        assert "n_detections" in out.columns
        assert "mean_m" in out.columns
        assert "p50_m" in out.columns
        assert "p90_m" in out.columns
        assert "max_m" in out.columns

    def test_lidar_stats_correct(self):
        out = distance_stats_by_modality(self._make_df())
        row = out[out["modality"] == "lidar"]
        assert len(row) == 1
        assert int(row["n_detections"].iloc[0]) == 3
        assert float(row["mean_m"].iloc[0]) == pytest.approx(20.0, abs=0.5)
        assert float(row["max_m"].iloc[0]) == 30.0

    def test_empty_modality_excluded(self):
        df = with_sensor_flags(_df([
            {"source": "radar", "distance_m": 40.0, "container_id": 1},
        ]))
        out = distance_stats_by_modality(df)
        assert "radar" in out["modality"].values
        assert "lidar" not in out["modality"].values
        assert "camera" not in out["modality"].values


# ── class_coverage_by_modality ───────────────────────────────────────────────


class TestClassCoverageByModality:
    def _make_df(self):
        df = _df([
            {"source": "lidar|camera", "detection_class": "car"},
            {"source": "lidar|camera", "detection_class": "car"},
            {"source": "camera", "detection_class": "pedestrian"},
            {"source": "radar", "detection_class": "car"},
        ])
        return with_sensor_flags(df)

    def test_returns_dataframe(self):
        out = class_coverage_by_modality(self._make_df())
        assert isinstance(out, pd.DataFrame)

    def test_modality_column_present(self):
        out = class_coverage_by_modality(self._make_df())
        assert "modality" in out.columns
        assert "detection_class" in out.columns
        assert "count" in out.columns

    def test_lidar_car_count(self):
        out = class_coverage_by_modality(self._make_df())
        row = out[(out["modality"] == "lidar") & (out["detection_class"] == "car")]
        assert len(row) == 1
        assert int(row["count"].iloc[0]) == 2

    def test_camera_pedestrian_count(self):
        out = class_coverage_by_modality(self._make_df())
        row = out[(out["modality"] == "camera") & (out["detection_class"] == "pedestrian")]
        assert len(row) == 1
        assert int(row["count"].iloc[0]) == 1

    def test_empty_input_returns_empty(self):
        empty = with_sensor_flags(
            pd.DataFrame({"source": pd.Series(dtype=str), "detection_class": pd.Series(dtype=str)})
        )
        out = class_coverage_by_modality(empty)
        assert len(out) == 0


# ── detection_gaps ───────────────────────────────────────────────────────────


class TestDetectionGaps:
    def _make_events(self):
        return _df([
            {"container_id": 1, "start_ts": 0, "end_ts": 1_000_000},
            {"container_id": 1, "start_ts": 2_000_000, "end_ts": 3_000_000},
        ])

    def test_returns_one_row_per_event_window(self):
        events = self._make_events()
        tracks = with_sensor_flags(_df([
            {"source": "lidar", "container_id": 1, "frame_ts": 500_000, "distance_m": 10.0},
        ]))
        out = detection_gaps(tracks, events)
        assert len(out) == 2

    def test_gap_columns_present(self):
        out = detection_gaps(with_sensor_flags(pd.DataFrame(
            columns=["source", "container_id", "frame_ts", "distance_m"]
        )), self._make_events())
        assert "lidar_gap" in out.columns
        assert "radar_gap" in out.columns

    def test_window_with_lidar_detections_is_not_a_lidar_gap(self):
        events = self._make_events()
        tracks = with_sensor_flags(_df([
            {"source": "lidar", "container_id": 1, "frame_ts": 500_000, "distance_m": 10.0},
        ]))
        out = detection_gaps(tracks, events)
        first_window = out[out["start_ts"] == 0].iloc[0]
        assert first_window["lidar_gap"] == False

    def test_window_without_radar_detections_is_a_radar_gap(self):
        events = self._make_events()
        tracks = with_sensor_flags(_df([
            {"source": "lidar", "container_id": 1, "frame_ts": 500_000, "distance_m": 10.0},
        ]))
        out = detection_gaps(tracks, events)
        first_window = out[out["start_ts"] == 0].iloc[0]
        assert first_window["radar_gap"] == True

    def test_empty_window_is_both_a_gap(self):
        events = self._make_events()
        tracks = with_sensor_flags(_df([
            {"source": "lidar", "container_id": 1, "frame_ts": 500_000, "distance_m": 10.0},
        ]))
        out = detection_gaps(tracks, events)
        second_window = out[out["start_ts"] == 2_000_000].iloc[0]
        assert second_window["lidar_gap"] == True
        assert second_window["radar_gap"] == True
        assert second_window["total_detections"] == 0

    def test_detection_counts_correct(self):
        events = _df([{"container_id": 1, "start_ts": 0, "end_ts": 2_000_000}])
        tracks = with_sensor_flags(_df([
            {"source": "lidar|camera", "container_id": 1, "frame_ts": 500_000, "distance_m": 10.0},
            {"source": "lidar|camera", "container_id": 1, "frame_ts": 1_000_000, "distance_m": 20.0},
            {"source": "radar", "container_id": 1, "frame_ts": 1_500_000, "distance_m": 30.0},
        ]))
        out = detection_gaps(tracks, events)
        row = out.iloc[0]
        assert row["total_detections"] == 3
        assert row["lidar_detections"] == 2
        assert row["radar_detections"] == 1
        assert row["lidar_gap"] == False
        assert row["radar_gap"] == False
