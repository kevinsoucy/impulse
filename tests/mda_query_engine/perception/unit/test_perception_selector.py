"""PerceptionSelector contract — `build`, AST round-trip, channel-side stub.

These tests don't need Spark: the selector reads a pandas DataFrame from
the cache and returns an Intervals object.
"""

import numpy as np
import pandas as pd

from mda_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesExpression,
)
from mda_query_engine.model.series.intervals import Intervals
from mda_query_engine.perception.tsal.perception_selector import (
    PerceptionCache,
    PerceptionSelector,
)


def _make_cache(rows):
    pdf = pd.DataFrame(rows)
    channels_pdf = pd.DataFrame(
        {
            "container_id": pd.Series([], dtype="int64"),
            "channel_id": pd.Series([], dtype="int64"),
            "tstart": pd.Series([], dtype="float64"),
            "tend": pd.Series([], dtype="float64"),
            "value": pd.Series([], dtype="float64"),
        }
    )
    col_map = {
        "cid": "container_id",
        "ch": "channel_id",
        "ts": "tstart",
        "te": "tend",
        "val": "value",
    }
    return PerceptionCache(
        channels_pdf=channels_pdf,
        col_map=col_map,
        object_tracks_pdf=pdf,
    )


class TestPerceptionSelectorBuild:
    def test_string_eq_predicate_matches_rows_and_returns_intervals(self):
        cache = _make_cache(
            [
                {"container_id": 1, "object_id": 10, "frame_ts": 100.0, "detection_class": "cyclist"},
                {"container_id": 1, "object_id": 10, "frame_ts": 200.0, "detection_class": "cyclist"},
                {"container_id": 1, "object_id": 11, "frame_ts": 100.0, "detection_class": "car"},
            ]
        )
        sel = PerceptionSelector("detection_class", "eq", "cyclist")
        result = sel.build(cache)
        assert isinstance(result, Intervals)
        assert len(result) >= 1
        assert result.start_time() == 100.0

    def test_numeric_lt_predicate_filters_correctly(self):
        cache = _make_cache(
            [
                {"container_id": 1, "object_id": 10, "frame_ts": 100.0, "distance_m": 5.0},
                {"container_id": 1, "object_id": 10, "frame_ts": 200.0, "distance_m": 12.0},
            ]
        )
        sel = PerceptionSelector("distance_m", "lt", 8.0)
        result = sel.build(cache)
        assert isinstance(result, Intervals)
        assert result.start_time() == 100.0

    def test_no_matching_rows_returns_empty_intervals(self):
        cache = _make_cache(
            [{"container_id": 1, "object_id": 10, "frame_ts": 100.0, "detection_class": "car"}]
        )
        sel = PerceptionSelector("detection_class", "eq", "cyclist")
        result = sel.build(cache)
        assert len(result) == 0

    def test_consecutive_matched_frames_for_same_object_merge_into_one_interval(self):
        cache = _make_cache(
            [
                {"container_id": 1, "object_id": 10, "frame_ts": 100.0, "detection_class": "cyclist"},
                {"container_id": 1, "object_id": 10, "frame_ts": 200.0, "detection_class": "cyclist"},
                {"container_id": 1, "object_id": 10, "frame_ts": 300.0, "detection_class": "cyclist"},
            ]
        )
        sel = PerceptionSelector("detection_class", "eq", "cyclist")
        result = sel.build(cache)
        assert len(result) == 1
        assert result.start_time() == 100.0
        assert result.end_time() >= 300.0

    def test_build_against_non_perception_cache_returns_empty(self):
        sel = PerceptionSelector("detection_class", "eq", "cyclist")
        from mda_query_engine.analyze.query.solvers.empty_cache import (
            EmptyTimeSeriesCache,
        )

        result = sel.build(EmptyTimeSeriesCache())
        assert isinstance(result, Intervals)
        assert len(result) == 0


class TestPerceptionSelectorChannelStub:
    def test_get_selector_expr_never_matches_channel_rows(self):
        # The selector_expr is what the 6-stage channel filter joins against.
        # Returning F.lit(False) is what prevents a PerceptionSelector from
        # accidentally matching a channel_metrics row. Verify it constructs
        # without error when called with an active SparkContext (the only
        # context in which get_selector_expr is invoked in production).
        #
        # Uses configure_spark_with_delta_pip so the JVM is initialized with
        # the Delta catalog jars — required when this test module runs before
        # any Spark-dependent tests that share the same JVM process.
        import pytest

        pytest.importorskip("pyspark")
        from delta import configure_spark_with_delta_pip
        from pyspark.sql import SparkSession
        from pyspark.sql.column import Column

        # getOrCreate() reuses any existing session (e.g. the root conftest's
        # session-scoped spark), so we must not call spark.stop() here — that
        # would destroy the shared session and break downstream tests.
        configure_spark_with_delta_pip(
            SparkSession.builder.master("local[1]")
            .config("spark.driver.host", "localhost")
        ).getOrCreate()
        sel = PerceptionSelector("detection_class", "eq", "cyclist")
        expr = sel.get_selector_expr()
        assert isinstance(expr, Column)
        # The expression renders as a boolean literal — no column name appears
        assert "false" in str(expr).lower() or "literal" in str(expr).lower()

    def test_required_tags_is_empty(self):
        sel = PerceptionSelector("detection_class", "eq", "cyclist")
        assert sel.required_tags() == set()
        assert sel.get_required_tag_exprs() == set()


class TestSourceContainsBuildBehavior:
    """source_contains("lidar") filters pipe-delimited source strings correctly."""

    def test_matches_rows_that_include_the_sensor(self):
        cache = _make_cache(
            [
                {"container_id": 1, "object_id": 10, "frame_ts": 100.0, "source": "lidar|camera"},
                {"container_id": 1, "object_id": 10, "frame_ts": 200.0, "source": "camera"},
                {"container_id": 1, "object_id": 10, "frame_ts": 300.0, "source": "lidar|radar|camera"},
            ]
        )
        sel = PerceptionSelector("source", "contains", "lidar")
        result = sel.build(cache)
        assert isinstance(result, Intervals)
        # Frames 100 and 300 include lidar; frame 200 (camera only) does not.
        assert result.start_time() == 100.0

    def test_excludes_rows_without_the_sensor(self):
        cache = _make_cache(
            [
                {"container_id": 1, "object_id": 10, "frame_ts": 100.0, "source": "camera"},
                {"container_id": 1, "object_id": 10, "frame_ts": 200.0, "source": "radar|camera"},
            ]
        )
        sel = PerceptionSelector("source", "contains", "lidar")
        result = sel.build(cache)
        assert len(result) == 0

    def test_multi_object_only_lidar_object_contributes_intervals(self):
        # Object 10 always has lidar; object 20 only has camera.
        # The per-container selector returns intervals for object 10 frames only.
        cache = _make_cache(
            [
                {"container_id": 1, "object_id": 10, "frame_ts": 100.0, "source": "lidar|camera"},
                {"container_id": 1, "object_id": 10, "frame_ts": 200.0, "source": "lidar|camera"},
                {"container_id": 1, "object_id": 20, "frame_ts": 100.0, "source": "camera"},
                {"container_id": 1, "object_id": 20, "frame_ts": 200.0, "source": "camera"},
            ]
        )
        sel = PerceptionSelector("source", "contains", "lidar")
        result = sel.build(cache)
        assert isinstance(result, Intervals)
        assert len(result) >= 1
        assert result.start_time() == 100.0

    def test_null_source_treated_as_non_matching(self):
        # None source fills to "" which does not contain "lidar" — should not crash
        # and should not contribute an interval.
        cache = _make_cache(
            [
                {"container_id": 1, "object_id": 10, "frame_ts": 100.0, "source": None},
                {"container_id": 1, "object_id": 10, "frame_ts": 200.0, "source": "lidar"},
            ]
        )
        sel = PerceptionSelector("source", "contains", "lidar")
        result = sel.build(cache)
        # Only frame 200 matches; frame 100 (None source) does not.
        assert result.start_time() == 200.0

    def test_non_contiguous_matching_frames_produce_separate_intervals(self):
        # LiDAR at frame 100, camera-only at 200 and 300, LiDAR again at 400.
        # The gap at 200-300 should produce two disjoint intervals, not one merged window.
        cache = _make_cache(
            [
                {"container_id": 1, "object_id": 10, "frame_ts": 100.0, "source": "lidar|camera"},
                {"container_id": 1, "object_id": 10, "frame_ts": 200.0, "source": "camera"},
                {"container_id": 1, "object_id": 10, "frame_ts": 300.0, "source": "camera"},
                {"container_id": 1, "object_id": 10, "frame_ts": 400.0, "source": "lidar|radar"},
            ]
        )
        sel = PerceptionSelector("source", "contains", "lidar")
        result = sel.build(cache)
        # Frame 100 → [100, 200); frame 400 → [400, 401).  The gap (200-400) breaks
        # these into two separate intervals.
        assert len(result) == 2
        assert result.start_time() == 100.0


class TestPerceptionSelectorASTRoundTrip:
    def test_as_dict_and_from_dict_preserve_predicate(self):
        sel = PerceptionSelector("detection_class", "eq", "cyclist")
        obj = sel.as_dict()
        restored = PerceptionSelector.from_dict(obj)
        assert restored.column == "cyclist".__class__("detection_class")  # str
        assert restored.column == "detection_class"
        assert restored.op == "eq"
        assert restored.value == "cyclist"

    def test_as_dict_includes_type_for_polymorphic_resolution(self):
        sel = PerceptionSelector("distance_m", "lt", 8.0)
        obj = sel.as_dict()
        assert "type" in obj
        assert obj["type"].endswith("PerceptionSelector")

    def test_round_trip_via_timeseriesexpression_from_dict(self):
        sel = PerceptionSelector("azimuth", "eq", "front_left")
        restored = TimeSeriesExpression.from_dict(sel.as_dict())
        assert isinstance(restored, PerceptionSelector)
        assert restored.column == "azimuth"
        assert restored.op == "eq"
        assert restored.value == "front_left"

    def test_get_selectors_returns_self(self):
        sel = PerceptionSelector("detection_class", "eq", "cyclist")
        selectors = sel.get_selectors()
        assert selectors == [sel]
