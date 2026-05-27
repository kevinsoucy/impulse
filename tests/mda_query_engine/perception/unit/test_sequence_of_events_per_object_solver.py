"""SequenceOfEvents over track-scoped perception steps — the
PerceptionSolver per-object UDF loop produces one chained window per
``(container_id, object_id)`` where the same object satisfies every step
in temporal order.

Exercises the per-container UDF (``_solve_perception_udf``) directly with
pandas inputs. End-to-end Spark coverage lives in the integration suite.
"""

import numpy as np
import pandas as pd
import pytest

from mda_reporting.events.sequence_of_events import SequenceOfEvents

from mda_query_engine.perception.query.perception_solver import PerceptionSolver
from mda_query_engine.perception.tsal.object_track_accessor import ObjectTrackAccessor


@pytest.fixture
def col_map():
    return {
        "cid": "container_id",
        "ch": "channel_id",
        "ts": "tstart",
        "te": "tend",
        "val": "value",
    }


@pytest.fixture
def empty_channels_pdf(col_map):
    return pd.DataFrame(
        {
            col_map["cid"]: pd.Series(dtype=np.int64),
            col_map["ch"]: pd.Series(dtype=np.int64),
            col_map["ts"]: pd.Series(dtype=np.float64),
            col_map["te"]: pd.Series(dtype=np.float64),
            col_map["val"]: pd.Series(dtype=np.float64),
        }
    )


def _lane_change_otp():
    """One object (10) cleanly transitions lane_offset −1 → 0; another (20)
    only ever sits at lane_offset = 1 so it never satisfies step 1. A
    correct track-scoped sequence solver returns exactly one chained
    window from object 10's transition.
    """
    return pd.DataFrame(
        [
            # Object 10 — transitions left lane to ego lane
            {"container_id": 1, "object_id": 10, "frame_ts": 100.0,
             "detection_class": "car", "azimuth": "front_left",
             "distance_m": 6.0, "confidence": 0.9, "source": "lidar",
             "lane_offset": -1, "relative_velocity_ms": 0.0},
            {"container_id": 1, "object_id": 10, "frame_ts": 200.0,
             "detection_class": "car", "azimuth": "front_left",
             "distance_m": 6.0, "confidence": 0.9, "source": "lidar",
             "lane_offset": -1, "relative_velocity_ms": 0.0},
            {"container_id": 1, "object_id": 10, "frame_ts": 300.0,
             "detection_class": "car", "azimuth": "front_left",
             "distance_m": 6.0, "confidence": 0.9, "source": "lidar",
             "lane_offset": 0, "relative_velocity_ms": 0.0},
            {"container_id": 1, "object_id": 10, "frame_ts": 400.0,
             "detection_class": "car", "azimuth": "front_left",
             "distance_m": 6.0, "confidence": 0.9, "source": "lidar",
             "lane_offset": 0, "relative_velocity_ms": 0.0},
            # Object 20 — always in right lane; only satisfies step 2's
            # value (lane_offset == 0) never, and step 1's value never.
            {"container_id": 1, "object_id": 20, "frame_ts": 100.0,
             "detection_class": "car", "azimuth": "front_right",
             "distance_m": 7.0, "confidence": 0.9, "source": "lidar",
             "lane_offset": 1, "relative_velocity_ms": 0.0},
            {"container_id": 1, "object_id": 20, "frame_ts": 200.0,
             "detection_class": "car", "azimuth": "front_right",
             "distance_m": 7.0, "confidence": 0.9, "source": "lidar",
             "lane_offset": 1, "relative_velocity_ms": 0.0},
        ]
    )


class TestTwoStepTrackScopedSequence:
    def test_emits_one_chained_window_keyed_by_object_id(
        self, col_map, empty_channels_pdf
    ):
        scoped = ObjectTrackAccessor()(track_scope=True)
        # Step 1: car with lane_offset == -1 (in left lane)
        # Step 2: same car with lane_offset == 0  (now in ego lane)
        seq = SequenceOfEvents(
            name="lane_change_left",
            expressions=[
                scoped.lane_offset == -1,
                scoped.lane_offset == 0,
            ],
            max_overlap=10_000.0,
        )
        # Apply alias the way Report registration would.
        selection = seq.get_expression()

        otp = _lane_change_otp()
        out_pdf = PerceptionSolver._solve_perception_udf(
            channels_pdf=empty_channels_pdf,
            object_tracks_pdf=otp,
            selections=[selection],
            col_map=col_map,
        )

        windows = out_pdf[seq.get_name()].iloc[0]
        # Only object 10 satisfies the sequence; object 20 never matches step 1.
        assert len(windows) == 1
        # Triple shape: [start, end, object_id]
        assert len(windows[0]) == 3
        assert int(windows[0][2]) == 10


def _full_row(container_id, object_id, frame_ts, detection_class, azimuth,
              distance_m=10.0, confidence=0.9, source="lidar",
              lane_offset=0, relative_velocity_ms=0.0):
    return {
        "container_id": container_id, "object_id": object_id,
        "frame_ts": frame_ts, "detection_class": detection_class,
        "azimuth": azimuth, "distance_m": distance_m,
        "confidence": confidence, "source": source,
        "lane_offset": lane_offset, "relative_velocity_ms": relative_velocity_ms,
    }


class TestPerceptionExpressionAsSequenceStep:
    """A PerceptionEvent's expression (string-column compound predicate) used
    directly as a step in SequenceOfEvents — mirrors the aeb_then_clear example
    in 03_authoring_events.md.

    Both steps here are perception expressions; the pattern being pinned is that
    string-column selectors (detection_class, azimuth) work correctly as
    SequenceOfEvents steps.

    SequenceOfEventsExpression rule: step N+1 must start at or after step N's
    start AND at or before step N's end (the two windows must overlap).
    """

    def _otp_car_then_pedestrian(self):
        """Car at frames 100–200; pedestrian overlaps from frame 150.

        Step 1 (car) interval: [100, 201].
        Step 2 (pedestrian & front_left) first interval starts at 150 ≤ 201
        and 150 ≥ 100, so the sequence fires → window [100, pedestrian-end].
        """
        return pd.DataFrame(
            [
                _full_row(1, 10, 100.0, "car", "front"),
                _full_row(1, 10, 200.0, "car", "front"),
                _full_row(1, 20, 150.0, "pedestrian", "front_left",
                          source="lidar|camera", lane_offset=1),
                _full_row(1, 20, 250.0, "pedestrian", "front_left",
                          source="lidar|camera", lane_offset=1),
            ]
        )

    def test_string_predicate_step_fires_sequence(
        self, col_map, empty_channels_pdf
    ):
        ot = ObjectTrackAccessor()
        seq = SequenceOfEvents(
            name="car_then_pedestrian",
            expressions=[
                ot.detection_class("car"),
                ot.detection_class("pedestrian") & ot.azimuth("front_left"),
            ],
        )
        selection = seq.get_expression()

        out_pdf = PerceptionSolver._solve_perception_udf(
            channels_pdf=empty_channels_pdf,
            object_tracks_pdf=self._otp_car_then_pedestrian(),
            selections=[selection],
            col_map=col_map,
        )

        windows = out_pdf[seq.get_name()].iloc[0]
        assert len(windows) >= 1
        # Non-scoped: pairs only.
        assert all(len(w) == 2 for w in windows)
        # Sequence starts at the first car frame.
        assert windows[0][0] == 100.0

    def test_reversed_step_order_does_not_fire(
        self, col_map, empty_channels_pdf
    ):
        # Pedestrian at frames 100-200, car not until 300.
        # Sequence is [car THEN pedestrian].
        # SequenceOfEventsExpression requires step2.start >= step1.start, so
        # a pedestrian interval starting at 100 (before car starts at 300)
        # does not satisfy that constraint → no fire.
        ot = ObjectTrackAccessor()
        seq = SequenceOfEvents(
            name="car_then_pedestrian_reversed",
            expressions=[
                ot.detection_class("car"),
                ot.detection_class("pedestrian"),
            ],
        )
        otp = pd.DataFrame(
            [
                _full_row(1, 20, 100.0, "pedestrian", "front_left"),
                _full_row(1, 20, 200.0, "pedestrian", "front_left"),
                _full_row(1, 10, 300.0, "car", "front"),
                _full_row(1, 10, 400.0, "car", "front"),
            ]
        )
        out_pdf = PerceptionSolver._solve_perception_udf(
            channels_pdf=empty_channels_pdf,
            object_tracks_pdf=otp,
            selections=[seq.get_expression()],
            col_map=col_map,
        )
        windows = out_pdf[seq.get_name()].iloc[0]
        assert len(windows) == 0

    def test_non_overlapping_steps_do_not_fire(
        self, col_map, empty_channels_pdf
    ):
        # Car at frames 100-200 (window ends ~201), pedestrian not until 500.
        # step2.start (500) > step1.end (201) → SequenceOfEventsExpression
        # condition next.start <= e_prev fails → no fire.
        ot = ObjectTrackAccessor()
        seq = SequenceOfEvents(
            name="car_then_ped_gap",
            expressions=[
                ot.detection_class("car"),
                ot.detection_class("pedestrian"),
            ],
        )
        otp = pd.DataFrame(
            [
                _full_row(1, 10, 100.0, "car", "front"),
                _full_row(1, 10, 200.0, "car", "front"),
                _full_row(1, 20, 500.0, "pedestrian", "front_left"),
                _full_row(1, 20, 600.0, "pedestrian", "front_left"),
            ]
        )
        out_pdf = PerceptionSolver._solve_perception_udf(
            channels_pdf=empty_channels_pdf,
            object_tracks_pdf=otp,
            selections=[seq.get_expression()],
            col_map=col_map,
        )
        windows = out_pdf[seq.get_name()].iloc[0]
        assert len(windows) == 0

    def test_closing_distance_single_object_fires_sequence(
        self, col_map, empty_channels_pdf
    ):
        # One object approaches: enters warning zone (< 20 m) then closes to
        # critical zone (< 8 m). Step 1 interval covers all three frames;
        # step 2 interval covers only the last frame (distance = 6 m).
        # Step 2 starts inside step 1 → sequence fires.
        ot = ObjectTrackAccessor()
        seq = SequenceOfEvents(
            name="approach_then_close",
            expressions=[
                ot.distance_m < 20.0,
                ot.distance_m < 8.0,
            ],
        )
        otp = pd.DataFrame(
            [
                _full_row(1, 10, 100.0, "car", "front", distance_m=15.0),
                _full_row(1, 10, 200.0, "car", "front", distance_m=12.0),
                _full_row(1, 10, 300.0, "car", "front", distance_m=6.0),
            ]
        )
        out_pdf = PerceptionSolver._solve_perception_udf(
            channels_pdf=empty_channels_pdf,
            object_tracks_pdf=otp,
            selections=[seq.get_expression()],
            col_map=col_map,
        )
        windows = out_pdf[seq.get_name()].iloc[0]
        assert len(windows) >= 1
        assert all(len(w) == 2 for w in windows)
        # Sequence starts at the first frame in the warning zone.
        assert windows[0][0] == 100.0


class TestNonScopedTwoStepSequenceFallsBackToPerContainer:
    def test_no_track_scope_emits_pairs(self, col_map, empty_channels_pdf):
        ot = ObjectTrackAccessor()
        seq = SequenceOfEvents(
            name="lane_change_anywhere",
            expressions=[
                ot.lane_offset == -1,
                ot.lane_offset == 0,
            ],
            max_overlap=10_000.0,
        )
        selection = seq.get_expression()

        otp = _lane_change_otp()
        out_pdf = PerceptionSolver._solve_perception_udf(
            channels_pdf=empty_channels_pdf,
            object_tracks_pdf=otp,
            selections=[selection],
            col_map=col_map,
        )

        windows = out_pdf[seq.get_name()].iloc[0]
        # Each window is just [start, end] — no object_id.
        for win in windows:
            assert len(win) == 2
