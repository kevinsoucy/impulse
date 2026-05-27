"""Complement (`~intervals`) returns the gaps against attached bounds.

Pins the contract for the new `__invert__` operator on `Intervals` —
bounds must be set on the Intervals object (either at construction or via
`with_bounds`) before complement is well-defined.
"""

import numpy as np
import pytest

from mda_query_engine.model.series.intervals import Intervals


class TestIntervalsInvert:
    def test_complement_of_single_middle_interval_splits_into_two(self):
        ivl = Intervals([3.0], [7.0], bounds=(0.0, 10.0))
        complement = ~ivl
        assert list(complement.tstarts) == [0.0, 7.0]
        assert list(complement.tends) == [3.0, 10.0]

    def test_complement_of_empty_intervals_is_full_bounds(self):
        ivl = Intervals(np.array([]), np.array([]), bounds=(0.0, 10.0))
        complement = ~ivl
        assert list(complement.tstarts) == [0.0]
        assert list(complement.tends) == [10.0]

    def test_complement_of_full_bounds_is_empty(self):
        ivl = Intervals([0.0], [10.0], bounds=(0.0, 10.0))
        complement = ~ivl
        assert len(complement) == 0

    def test_complement_of_multiple_intervals_returns_interleaved_gaps(self):
        ivl = Intervals([1.0, 5.0], [3.0, 7.0], bounds=(0.0, 10.0))
        complement = ~ivl
        assert list(complement.tstarts) == [0.0, 3.0, 7.0]
        assert list(complement.tends) == [1.0, 5.0, 10.0]

    def test_complement_drops_zero_length_pieces_at_bounds_endpoints(self):
        ivl = Intervals([0.0], [10.0], bounds=(0.0, 10.0))
        complement = ~ivl
        assert len(complement) == 0

    def test_complement_without_bounds_raises(self):
        ivl = Intervals([2.0], [4.0])
        with pytest.raises(ValueError, match="bounds"):
            ~ivl

    def test_with_bounds_attaches_bounds_without_changing_data(self):
        ivl = Intervals([2.0], [4.0])
        rebound = ivl.with_bounds((0.0, 10.0))
        assert list(rebound.tstarts) == [2.0]
        assert list(rebound.tends) == [4.0]
        complement = ~rebound
        assert list(complement.tstarts) == [0.0, 4.0]
        assert list(complement.tends) == [2.0, 10.0]

    def test_double_complement_returns_original(self):
        original = Intervals([3.0], [7.0], bounds=(0.0, 10.0))
        twice = ~(~original)
        assert list(twice.tstarts) == [3.0]
        assert list(twice.tends) == [7.0]

    def test_complement_propagates_bounds(self):
        ivl = Intervals([3.0], [7.0], bounds=(0.0, 10.0))
        complement = ~ivl
        assert complement.bounds == (0.0, 10.0)

    def test_existing_and_or_unaffected_by_bounds_change(self):
        a = Intervals([1.0, 5.0], [3.0, 7.0])
        b = Intervals([2.0, 6.0], [4.0, 8.0])
        intersection = a & b
        union = a | b
        assert list(intersection.tstarts) == [2.0, 6.0]
        assert list(intersection.tends) == [3.0, 7.0]
        assert list(union.tstarts) == [1.0, 5.0]
        assert list(union.tends) == [4.0, 8.0]


class TestIntervalsBackwardsCompatible:
    """Acceptance signal #3 (no-regression on the core API).

    The new ``bounds`` kwarg on ``Intervals.__init__`` is keyword-only and
    optional; all existing call sites continue to work unchanged.  These
    tests run without Spark and complement the Spark-dependent
    ``test_basic_event_no_regression.py`` in the integration suite.
    """

    def test_construct_without_bounds_still_works(self):
        ivl = Intervals([1.0], [3.0])
        assert ivl.bounds is None
        assert len(ivl) == 1

    def test_merge_overlaps_still_works(self):
        ivl = Intervals([1.0, 2.0], [3.0, 4.0], merge_overlaps=True)
        assert list(ivl.tstarts) == [1.0]
        assert list(ivl.tends) == [4.0]

    def test_empty_intervals_factory_unchanged(self):
        empty = Intervals.empty()
        assert len(empty) == 0
        assert empty.bounds is None
