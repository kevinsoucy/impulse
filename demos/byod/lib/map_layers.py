"""Vectorized nuScenes map-expansion layers as queryable shapely geometry.

Static map geometry is **not** a time-series — it is a spatial lookup. This module
turns one location's map-expansion layers (`expansion/<location>.json`) into shapely
geometries indexed for fast point queries, so the nuScenes adapter can derive a
**per-keyframe `map_context` series** — the actual time-series artifact — by joining
ego / object positions against the map at each frame.

Layer JSON shape (map-expansion v1.3):
- ``node``    : {token, x, y}                       — vertices in the location's global frame
- ``polygon`` : {token, exterior_node_tokens, holes}— holes = [{node_tokens: [...]}]
- layers (``ped_crossing``, ``walkway``, ``stop_line``, ``lane``, ``road_segment``)
  reference a ``polygon_token``; ``drivable_area`` references ``polygon_tokens`` (many);
  ``road_segment`` carries ``is_intersection``.

Coordinates are in the same global frame as nuScenes ego-pose and annotation
translations, so a position ``(x, y)`` queries the map directly with no transform.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree


def _build_polygon(poly_rec: dict, node_xy: dict[str, tuple[float, float]]) -> Polygon | None:
    """Build a shapely Polygon (with holes) from a polygon record, or None if degenerate."""
    exterior = [node_xy[t] for t in poly_rec["exterior_node_tokens"] if t in node_xy]
    if len(exterior) < 3:
        return None
    holes = []
    for hole in poly_rec.get("holes", []):
        ring = [node_xy[t] for t in hole.get("node_tokens", []) if t in node_xy]
        if len(ring) >= 3:
            holes.append(ring)
    try:
        return Polygon(exterior, holes)
    except Exception:  # malformed ring — skip rather than fail the whole map
        return None


class _GeomIndex:
    """A set of geometries with an STRtree for `contains` / nearest-distance queries.

    Optionally carries a parallel list of ids (used by the lane layer so a hit can
    report *which* lane). Empty layers answer False / None without a tree.
    """

    def __init__(self, geoms: list, ids: list[str] | None = None) -> None:
        self._geoms = geoms
        self._ids = ids
        self._tree = STRtree(geoms) if geoms else None

    def contains(self, x: float, y: float) -> bool:
        if self._tree is None:
            return False
        pt = Point(x, y)
        # predicate="intersects" → indices whose geometry truly contains/touches pt,
        # not merely whose bounding box overlaps.
        return len(self._tree.query(pt, predicate="intersects")) > 0

    def nearest_distance(self, x: float, y: float) -> float | None:
        """Distance to the nearest geometry (0.0 if the point is inside one).
        None if the layer has no geometries."""
        if self._tree is None:
            return None
        pt = Point(x, y)
        idx = self._tree.nearest(pt)
        return float(self._geoms[idx].distance(pt))

    def id_at(self, x: float, y: float) -> str | None:
        """Id of the geometry containing the point (first match), or None."""
        if self._tree is None or self._ids is None:
            return None
        pt = Point(x, y)
        hits = self._tree.query(pt, predicate="intersects")
        return self._ids[hits[0]] if len(hits) > 0 else None


# Module-level cache: parsing + indexing a location is expensive; one location's
# map serves every scene recorded there. Keyed by (expansion_dir, location).
_CACHE: dict[tuple[str, str], "MapLayers"] = {}


class MapLayers:
    """Queryable map layers for one nuScenes location.

    Build with :meth:`load` (cached). Query methods take a position ``(x, y)`` in the
    location's global frame and return per-frame map-context features.
    """

    def __init__(
        self,
        location: str,
        ped_crossing: _GeomIndex,
        walkway: _GeomIndex,
        drivable_area: _GeomIndex,
        intersection: _GeomIndex,
        stop_line: _GeomIndex,
        lane: _GeomIndex,
    ) -> None:
        self.location = location
        self._ped_crossing = ped_crossing
        self._walkway = walkway
        self._drivable_area = drivable_area
        self._intersection = intersection
        self._stop_line = stop_line
        self._lane = lane

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, expansion_dir: str | Path, location: str) -> "MapLayers":
        """Load and index a location's layers from ``<expansion_dir>/<location>.json``.

        Cached per (expansion_dir, location)."""
        key = (str(expansion_dir), location)
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        layers = cls._parse(Path(expansion_dir) / f"{location}.json", location)
        _CACHE[key] = layers
        return layers

    @classmethod
    def _parse(cls, path: Path, location: str) -> "MapLayers":
        with open(path) as f:
            raw = json.load(f)

        node_xy = {n["token"]: (n["x"], n["y"]) for n in raw["node"]}
        poly_by_token = {p["token"]: p for p in raw["polygon"]}

        def polys_for(records: list[dict], token_key: str = "polygon_token") -> list:
            geoms = []
            for rec in records:
                tok = rec[token_key]
                poly = poly_by_token.get(tok)
                if poly is None:
                    continue
                g = _build_polygon(poly, node_xy)
                if g is not None:
                    geoms.append(g)
            return geoms

        # drivable_area: one (or few) records, each with many polygon_tokens.
        drivable_geoms = []
        for rec in raw.get("drivable_area", []):
            for tok in rec.get("polygon_tokens", []):
                poly = poly_by_token.get(tok)
                if poly is not None:
                    g = _build_polygon(poly, node_xy)
                    if g is not None:
                        drivable_geoms.append(g)

        # intersection: road_segments flagged is_intersection.
        intersection_recs = [r for r in raw.get("road_segment", []) if r.get("is_intersection")]

        # lanes: keep the lane token as the id so a hit reports which lane.
        lane_recs = raw.get("lane", [])
        lane_geoms, lane_ids = [], []
        for rec in lane_recs:
            poly = poly_by_token.get(rec["polygon_token"])
            if poly is None:
                continue
            g = _build_polygon(poly, node_xy)
            if g is not None:
                lane_geoms.append(g)
                lane_ids.append(rec["token"])

        return cls(
            location=location,
            ped_crossing=_GeomIndex(polys_for(raw.get("ped_crossing", []))),
            walkway=_GeomIndex(polys_for(raw.get("walkway", []))),
            drivable_area=_GeomIndex(drivable_geoms),
            intersection=_GeomIndex(polys_for(intersection_recs)),
            stop_line=_GeomIndex(polys_for(raw.get("stop_line", []))),
            lane=_GeomIndex(lane_geoms, ids=lane_ids),
        )

    # ── Queries (position in the location's global frame) ──────────────────────

    def on_ped_crossing(self, x: float, y: float) -> bool:
        return self._ped_crossing.contains(x, y)

    def on_walkway(self, x: float, y: float) -> bool:
        return self._walkway.contains(x, y)

    def on_drivable_area(self, x: float, y: float) -> bool:
        return self._drivable_area.contains(x, y)

    def in_intersection(self, x: float, y: float) -> bool:
        return self._intersection.contains(x, y)

    def dist_to_ped_crossing_m(self, x: float, y: float) -> float | None:
        return self._ped_crossing.nearest_distance(x, y)

    def dist_to_stop_line_m(self, x: float, y: float) -> float | None:
        return self._stop_line.nearest_distance(x, y)

    def lane_id_at(self, x: float, y: float) -> str | None:
        return self._lane.id_at(x, y)
