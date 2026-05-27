"""Acceptance signal #2: ``~ot.detection_class("cyclist")`` returns the
complement intervals against container bounds.

The complement uses the bounds attached to the ``PerceptionCache`` when the
solver UDF constructs it — this test pins the contract that the bounds are
threaded through the build step.
"""

import numpy as np
import pandas as pd

from mda_query_engine.model.series.intervals import Intervals
from lakevision.tsal.object_track_accessor import ObjectTrackAccessor
from lakevision.tsal.perception_selector import PerceptionCache


def _cache_with_one_cyclist_window():
    container_bounds = (0.0, 1000.0)
    channels_pdf = pd.DataFrame(
        {
            "container_id": pd.Series([], dtype="int64"),
            "channel_id": pd.Series([], dtype="int64"),
            "tstart": pd.Series([], dtype="float64"),
            "tend": pd.Series([], dtype="float64"),
            "value": pd.Series([], dtype="float64"),
        }
    )
    object_tracks_pdf = pd.DataFrame(
        [
            {"container_id": 1, "object_id": 10, "frame_ts": 300.0,
             "detection_class": "cyclist", "azimuth": "front",
             "distance_m": 5.0, "confidence": 0.9, "source": "lidar"},
            {"container_id": 1, "object_id": 10, "frame_ts": 400.0,
             "detection_class": "cyclist", "azimuth": "front",
             "distance_m": 5.0, "confidence": 0.9, "source": "lidar"},
            # Non-cyclist row — irrelevant to the predicate
            {"container_id": 1, "object_id": 11, "frame_ts": 500.0,
             "detection_class": "car", "azimuth": "rear",
             "distance_m": 8.0, "confidence": 0.9, "source": "lidar"},
        ]
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
        object_tracks_pdf=object_tracks_pdf,
        container_bounds=container_bounds,
    )


class TestPerceptionInvert:
    def test_invert_cyclist_returns_complement_against_container_bounds(self):
        cache = _cache_with_one_cyclist_window()
        ot = ObjectTrackAccessor()
        complement = (~ot.detection_class("cyclist")).alias("not_cyclist").build(cache)
        assert isinstance(complement, Intervals)
        # Container bounds are (0, 1000); cyclist window spans roughly [300, 401].
        # Complement should bracket the cyclist window with the container bounds.
        assert complement.start_time() == 0.0
        assert complement.end_time() == 1000.0
        # At least two gap intervals: the pre-cyclist [0, 300] and post-cyclist
        # [end_of_cyclist, 1000].
        assert len(complement) >= 2

    def test_invert_propagates_bounds_so_double_invert_round_trips(self):
        cache = _cache_with_one_cyclist_window()
        ot = ObjectTrackAccessor()
        original = ot.detection_class("cyclist").alias("cyclist").build(cache)
        twice = (~(~ot.detection_class("cyclist"))).alias("twice").build(cache)
        # build() runs the AST top-down. Each application of __invert__ in the
        # AST materialises a complement at build time. The double-complement
        # should return the original windows.
        assert list(twice.tstarts) == list(original.tstarts)
        assert list(twice.tends) == list(original.tends)

    def test_invert_on_predicate_with_no_matches_yields_full_bounds(self):
        cache = _cache_with_one_cyclist_window()
        ot = ObjectTrackAccessor()
        # No truck in the data — invert should return the whole window.
        complement = (~ot.detection_class("truck")).alias("not_truck").build(cache)
        assert list(complement.tstarts) == [0.0]
        assert list(complement.tends) == [1000.0]
