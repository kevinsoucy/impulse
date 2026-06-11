"""``event_instance_id`` generation: deterministic per (container, event_name,
window[, entity_key]), so re-running a report reproduces the same ids and two
distinct entities in the same window get distinct ids.

The id is a CRC32 over ``container_id::event_name::start_ts::end_ts[::entity_key]``
— it does not encode the event's definition, so a redefinition shifts the id only
insofar as it changes the matched windows.
"""

import pyspark.sql.types as T

from impulse_reporting.util.event_instance_util import generate_event_instance_id_column

_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType()),
        T.StructField("event_name", T.StringType()),
        T.StructField("start_ts", T.LongType()),
        T.StructField("end_ts", T.LongType()),
        T.StructField("entity_key", T.StringType()),
    ]
)


def _ids(df, **kwargs):
    return [
        r.event_instance_id
        for r in df.withColumn(
            "event_instance_id", generate_event_instance_id_column(**kwargs)
        ).collect()
    ]


def test_event_instance_id_is_deterministic_across_builds(spark):
    df = spark.createDataFrame([(1, "near", 0, 10, None)], _SCHEMA)
    assert _ids(df) == _ids(df)  # pure CRC32 → same inputs reproduce the same id


def test_event_instance_id_distinct_per_window(spark):
    # Same event/container, different windows → different ids (so a re-run that
    # produces a new window does not collide with the old one).
    df = spark.createDataFrame([(1, "near", 0, 10, None), (1, "near", 0, 20, None)], _SCHEMA)
    a, b = _ids(df)
    assert a != b


def test_entity_key_folding_distinguishes_entities_in_same_window(spark):
    # With entity_key_col set, two entities matching the *same* window of the same
    # event get distinct ids; without folding they would collide.
    df = spark.createDataFrame(
        [
            (1, "near", 0, 10, '{"object_tracks": {"lidar": ["47"]}}'),
            (1, "near", 0, 10, '{"object_tracks": {"lidar": ["48"]}}'),
        ],
        _SCHEMA,
    )
    folded = _ids(df, entity_key_col="entity_key")
    assert folded[0] != folded[1]
    # Without folding the two rows are indistinguishable → identical ids.
    unfolded = _ids(df)
    assert unfolded[0] == unfolded[1]


def test_entity_key_none_preserves_legacy_hash(spark):
    # The default (entity_key_col=None) leaves the hash unchanged, so existing
    # event types keep their 1.0 ids — folding only applies when explicitly asked.
    df = spark.createDataFrame([(1, "near", 0, 10, "ignored")], _SCHEMA)
    assert _ids(df, entity_key_col=None) == _ids(df)
