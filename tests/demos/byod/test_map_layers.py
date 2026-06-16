"""Unit tests for demos/byod/lib/map_layers.py.

Builds a tiny synthetic map-expansion JSON (squares) and exercises the real parse +
query path: point-in-layer, nearest-distance, lane id, intersection, empty layers,
and the per-(dir, location) cache.
"""

import json

import pytest

from lib.map_layers import MapLayers


def _square(x0, y0, x1, y1):
    """Four corner (x, y) tuples of an axis-aligned square."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _build_raw(*, with_ped_crossing=True):
    """Assemble a minimal map-expansion 'expansion/<location>.json' structure.

    Layout (meters):
      ped_crossing  square (0,0)-(2,2)
      walkway       square (5,0)-(7,2)
      stop_line     square (3,3)-(4,4)
      lane "lane-1" square (0,5)-(2,7)
      road_segment  square (10,10)-(14,14)  is_intersection=True
      road_segment  square (20,20)-(22,22)  is_intersection=False
      drivable_area square (-1,-1)-(30,30)  (covers everything above)
    """
    nodes = []
    polygons = []
    node_tok = [0]
    poly_tok = [0]

    def add_polygon(corners):
        node_tokens = []
        for (x, y) in corners:
            t = f"n{node_tok[0]}"
            node_tok[0] += 1
            nodes.append({"token": t, "x": float(x), "y": float(y)})
            node_tokens.append(t)
        pt = f"p{poly_tok[0]}"
        poly_tok[0] += 1
        polygons.append({"token": pt, "exterior_node_tokens": node_tokens, "holes": []})
        return pt

    ped_poly = add_polygon(_square(0, 0, 2, 2))
    walk_poly = add_polygon(_square(5, 0, 7, 2))
    stop_poly = add_polygon(_square(3, 3, 4, 4))
    lane_poly = add_polygon(_square(0, 5, 2, 7))
    inter_poly = add_polygon(_square(10, 10, 14, 14))
    noninter_poly = add_polygon(_square(20, 20, 22, 22))
    drivable_poly = add_polygon(_square(-1, -1, 30, 30))

    return {
        "version": "1.3",
        "node": nodes,
        "polygon": polygons,
        "ped_crossing": (
            [{"token": "pc1", "polygon_token": ped_poly}] if with_ped_crossing else []
        ),
        "walkway": [{"token": "wk1", "polygon_token": walk_poly}],
        "stop_line": [{"token": "sl1", "polygon_token": stop_poly, "stop_line_type": "STOP_SIGN"}],
        "lane": [{"token": "lane-1", "polygon_token": lane_poly, "lane_type": "CAR"}],
        "road_segment": [
            {"token": "rs1", "polygon_token": inter_poly, "is_intersection": True},
            {"token": "rs2", "polygon_token": noninter_poly, "is_intersection": False},
        ],
        "drivable_area": [{"token": "da1", "polygon_tokens": [drivable_poly]}],
    }


@pytest.fixture
def layers(tmp_path):
    raw = _build_raw()
    (tmp_path / "testtown.json").write_text(json.dumps(raw))
    # Fresh cache key per tmp_path, so tests don't cross-contaminate.
    return MapLayers.load(tmp_path, "testtown")


class TestMembership:
    def test_inside_ped_crossing(self, layers):
        assert layers.on_ped_crossing(1.0, 1.0) is True
        assert layers.dist_to_ped_crossing_m(1.0, 1.0) == 0.0

    def test_outside_ped_crossing(self, layers):
        assert layers.on_ped_crossing(100.0, 100.0) is False
        assert layers.dist_to_ped_crossing_m(100.0, 100.0) > 0.0

    def test_on_walkway(self, layers):
        assert layers.on_walkway(6.0, 1.0) is True
        assert layers.on_walkway(1.0, 1.0) is False

    def test_on_drivable_area(self, layers):
        assert layers.on_drivable_area(1.0, 1.0) is True
        assert layers.on_drivable_area(1000.0, 1000.0) is False

    def test_in_intersection_true_only_for_flagged_segment(self, layers):
        assert layers.in_intersection(12.0, 12.0) is True   # rs1, is_intersection
        assert layers.in_intersection(21.0, 21.0) is False  # rs2, not flagged

    def test_boundary_point_counts_as_on(self, layers):
        # A point exactly on the ped_crossing edge intersects the polygon.
        assert layers.on_ped_crossing(0.0, 1.0) is True


class TestDistances:
    def test_nearest_stop_line(self, layers):
        # Nearest stop_line corner to (1,1) is (3,3): distance sqrt(8) ≈ 2.828.
        assert layers.dist_to_stop_line_m(1.0, 1.0) == pytest.approx(2.828, abs=1e-2)

    def test_distance_zero_when_inside(self, layers):
        assert layers.dist_to_stop_line_m(3.5, 3.5) == 0.0


class TestLaneId:
    def test_lane_id_inside(self, layers):
        assert layers.lane_id_at(1.0, 6.0) == "lane-1"

    def test_lane_id_none_outside_any_lane(self, layers):
        assert layers.lane_id_at(1.0, 1.0) is None  # inside ped_crossing, not a lane


class TestEmptyLayer:
    def test_empty_ped_crossing_layer(self, tmp_path):
        raw = _build_raw(with_ped_crossing=False)
        (tmp_path / "emptytown.json").write_text(json.dumps(raw))
        m = MapLayers.load(tmp_path, "emptytown")
        assert m.on_ped_crossing(1.0, 1.0) is False
        # No geometries → nearest distance is undefined, reported as None.
        assert m.dist_to_ped_crossing_m(1.0, 1.0) is None


class TestCache:
    def test_load_is_cached_per_dir_location(self, tmp_path):
        raw = _build_raw()
        (tmp_path / "cachetown.json").write_text(json.dumps(raw))
        a = MapLayers.load(tmp_path, "cachetown")
        b = MapLayers.load(tmp_path, "cachetown")
        assert a is b
