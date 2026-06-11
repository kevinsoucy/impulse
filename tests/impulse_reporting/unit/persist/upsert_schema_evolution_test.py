# pylint: disable=missing-function-docstring
"""The MERGE (upsert) path must evolve an existing table's schema for an additive
nullable column.

This is the 1.0 -> 2.0 upgrade guard. In 2.0 every event fact row carries the new
nullable `entity_key` column (NULL for BasicEvent / SequenceOfEvents). A 1.0
user's existing `event_instance_fact` table has no such column. The
unchanged-definition persist path uses ``UnityCatalogSink.upsert`` (Delta MERGE),
which previously did not enable schema evolution — so the first 2.0 write to a
legacy table would fail with a schema mismatch. ``upsert`` now calls
``withSchemaEvolution()`` (when ``overwrite_schema`` is set, the default), adding
the column to the existing table instead of raising.
"""

import pyspark.sql.types as T

from impulse_reporting.persist.report_storage import UnityCatalogSink, UnitySinkConfig

# 1.0-shaped event fact table — no entity_key column.
_LEGACY_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.IntegerType(), False),
        T.StructField("event_instance_id", T.LongType(), False),
        T.StructField("event_id", T.IntegerType(), False),
        T.StructField("start_ts", T.LongType(), False),
        T.StructField("end_ts", T.LongType(), False),
    ]
)
# A genuine 1.0-on-disk table: event_instance_id was IntegerType before crc32
# outgrew int32 and it was widened to LongType in 2.0. The upgrade MERGE must
# widen the existing column, not downcast the new (long) source values into it.
_LEGACY_INT_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.IntegerType(), False),
        T.StructField("event_instance_id", T.IntegerType(), False),
        T.StructField("event_id", T.IntegerType(), False),
        T.StructField("start_ts", T.LongType(), False),
        T.StructField("end_ts", T.LongType(), False),
    ]
)
# 2.0-shaped rows — adds the nullable entity_key.
_NEW_SCHEMA = T.StructType(
    [*_LEGACY_SCHEMA.fields, T.StructField("entity_key", T.StringType(), True)]
)

_MERGE_KEYS = ["container_id", "event_id", "event_instance_id"]
# The production event-fact merge key in 2.0 (report.py) — entity_key included.
_ENTITY_MERGE_KEYS = [*_MERGE_KEYS, "entity_key"]
_URI = "spark_catalog.gold.upsert_schema_evolution_test"


def test_upsert_evolves_legacy_table_with_new_nullable_column(spark):
    spark.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.gold")
    spark.sql(f"DROP TABLE IF EXISTS {_URI}")

    # A pre-existing 1.0 table with one row and no entity_key column.
    spark.createDataFrame([(1, 100, 7, 0, 10)], _LEGACY_SCHEMA).write.format("delta").saveAsTable(
        _URI
    )
    assert "entity_key" not in spark.table(_URI).columns

    sink = UnityCatalogSink(
        UnitySinkConfig(catalog_name="spark_catalog", schema_name="gold", table_prefix="")
    )
    # 2.0 rows: one updates the existing key (NULL entity_key, like a 1.0 event),
    # one is a new EntityEvent row carrying a populated entity_key.
    new_rows = spark.createDataFrame(
        [
            (1, 100, 7, 0, 10, None),
            (2, 200, 8, 5, 15, '{"object_tracks": {"lidar": ["47"]}}'),
        ],
        _NEW_SCHEMA,
    )

    # Before the fix this raised a schema-mismatch on the MERGE.
    sink.upsert(new_rows, _URI, merge_keys=_MERGE_KEYS)

    result = spark.table(_URI)
    assert "entity_key" in result.columns  # column added to the legacy table
    by_cid = {r.container_id: r for r in result.collect()}
    assert set(by_cid) == {1, 2}
    assert by_cid[1].entity_key is None  # legacy-style row stays NULL
    assert by_cid[2].entity_key == '{"object_tracks": {"lidar": ["47"]}}'  # new row inserted

    spark.sql(f"DROP TABLE IF EXISTS {_URI}")


def test_upsert_without_overwrite_schema_does_not_evolve_legacy_table(spark):
    # Complement of the guard above: with overwrite_schema=False the MERGE skips
    # withSchemaEvolution(), so a legacy table is NOT evolved — the new entity_key
    # column is silently dropped (the row still inserts, without it). This is why
    # overwrite_schema defaults to True: it is what retains entity_key across the
    # 1.0 -> 2.0 upgrade.
    uri = "spark_catalog.gold.upsert_no_schema_evolution_test"
    spark.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.gold")
    spark.sql(f"DROP TABLE IF EXISTS {uri}")
    spark.createDataFrame([(1, 100, 7, 0, 10)], _LEGACY_SCHEMA).write.format("delta").saveAsTable(
        uri
    )

    sink = UnityCatalogSink(
        UnitySinkConfig(catalog_name="spark_catalog", schema_name="gold", table_prefix="")
    )
    new_rows = spark.createDataFrame(
        [(2, 200, 8, 5, 15, '{"object_tracks": {"lidar": ["47"]}}')], _NEW_SCHEMA
    )

    sink.upsert(new_rows, uri, merge_keys=_MERGE_KEYS, overwrite_schema=False)

    result = spark.table(uri)
    assert "entity_key" not in result.columns  # legacy schema left unevolved
    assert result.count() == 2  # row still inserts, just without entity_key
    spark.sql(f"DROP TABLE IF EXISTS {uri}")


def test_upsert_widens_legacy_int_event_instance_id_to_long(spark):
    # 1.0 -> 2.0 upgrade against a real 1.0 table whose event_instance_id is INT.
    # withSchemaEvolution() only ADDS columns; it does not widen an existing
    # column's type, so without the reconciliation step the long source values
    # would downcast into the int32 column and overflow for any crc32 > 2**31-1.
    uri = "spark_catalog.gold.upsert_int_widening_test"
    spark.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.gold")
    spark.sql(f"DROP TABLE IF EXISTS {uri}")

    spark.createDataFrame([(1, 100, 7, 0, 10)], _LEGACY_INT_SCHEMA).write.format(
        "delta"
    ).saveAsTable(uri)
    assert dict(spark.table(uri).dtypes)["event_instance_id"] == "int"

    sink = UnityCatalogSink(
        UnitySinkConfig(catalog_name="spark_catalog", schema_name="gold", table_prefix="")
    )
    big_id = 3_000_000_000  # > 2**31 - 1 (2_147_483_647): a realistic crc32 value
    new_rows = spark.createDataFrame(
        [
            (1, 100, 7, 0, 10, None),  # updates the existing key
            (2, big_id, 8, 5, 15, '{"object_tracks": {"lidar": ["47"]}}'),  # large id
        ],
        _NEW_SCHEMA,
    )

    sink.upsert(new_rows, uri, merge_keys=_MERGE_KEYS)

    result = spark.table(uri)
    # Column widened to bigint, and the > int32 value round-trips intact.
    assert dict(result.dtypes)["event_instance_id"] == "bigint"
    by_cid = {r.container_id: r for r in result.collect()}
    assert set(by_cid) == {1, 2}
    assert by_cid[2].event_instance_id == big_id
    # The pre-existing int row survives the widening rewrite unchanged.
    assert by_cid[1].event_instance_id == 100
    # The source-introduced entity_key column was also added by reconciliation.
    assert "entity_key" in result.columns
    spark.sql(f"DROP TABLE IF EXISTS {uri}")


def test_upsert_mixed_basic_and_entity_rows_stay_distinct(spark):
    # event_instance_fact is shared: BasicEvent rows carry entity_key=NULL,
    # EntityEvent rows carry the JSON map. A basic row and an entity row that
    # collide on (container_id, event_id, event_instance_id) must stay separate
    # (a NULL entity_key never matches a populated one), and each must upsert
    # idempotently on re-run — which requires the NULL-safe `<=>` so the basic
    # row's NULL matches its own NULL instead of duplicating.
    uri = "spark_catalog.gold.upsert_mixed_basic_entity_test"
    spark.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.gold")
    spark.sql(f"DROP TABLE IF EXISTS {uri}")

    sink = UnityCatalogSink(
        UnitySinkConfig(catalog_name="spark_catalog", schema_name="gold", table_prefix="")
    )
    entity_key = '{"object_tracks": {"lidar": ["47"]}}'
    basic_row = (1, 700, 7, 0, 10, None)
    entity_row = (1, 700, 7, 0, 10, entity_key)

    sink.upsert(
        spark.createDataFrame([basic_row, entity_row], _NEW_SCHEMA), uri, _ENTITY_MERGE_KEYS
    )
    assert spark.table(uri).count() == 2  # collide on the id triple, kept distinct

    # Re-run: each row updates in place; the NULL-entity_key basic row must not
    # duplicate (it would under plain `=`, since NULL = NULL is never true).
    sink.upsert(
        spark.createDataFrame([basic_row, entity_row], _NEW_SCHEMA), uri, _ENTITY_MERGE_KEYS
    )
    result = spark.table(uri)
    assert result.count() == 2
    assert {r.entity_key for r in result.collect()} == {None, entity_key}
    spark.sql(f"DROP TABLE IF EXISTS {uri}")


def test_is_safe_widening_only_widens_within_a_numeric_family():
    # The reconciliation helper must widen only same-family numeric promotions
    # and never narrow or cross families (that would corrupt or wrongly rewrite
    # an existing column).
    widen = UnityCatalogSink._is_safe_widening

    # Same family, strictly wider → allowed.
    assert widen(T.IntegerType(), T.LongType())
    assert widen(T.ByteType(), T.IntegerType())
    assert widen(T.ShortType(), T.LongType())
    assert widen(T.FloatType(), T.DoubleType())

    # Same type → not a widening (no rewrite needed).
    assert not widen(T.LongType(), T.LongType())
    assert not widen(T.DoubleType(), T.DoubleType())

    # Narrowing → refused.
    assert not widen(T.LongType(), T.IntegerType())
    assert not widen(T.DoubleType(), T.FloatType())

    # Cross-family (int<->float, or to/from non-numeric) → refused.
    assert not widen(T.IntegerType(), T.DoubleType())
    assert not widen(T.LongType(), T.DoubleType())
    assert not widen(T.IntegerType(), T.StringType())
    assert not widen(T.StringType(), T.StringType())


def test_upsert_entity_key_merge_key_keeps_colliding_entities_distinct(spark):
    # event_instance_id folds entity_key into a 32-bit crc32, so two distinct
    # entities can collide on (container_id, event_id, event_instance_id).
    # Including entity_key in the merge key keeps them as separate rows instead
    # of one silently overwriting the other.
    uri = "spark_catalog.gold.upsert_entity_collision_test"
    spark.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.gold")
    spark.sql(f"DROP TABLE IF EXISTS {uri}")

    sink = UnityCatalogSink(
        UnitySinkConfig(catalog_name="spark_catalog", schema_name="gold", table_prefix="")
    )
    key_47 = '{"object_tracks": {"lidar": ["47"]}}'
    key_88 = '{"object_tracks": {"lidar": ["88"]}}'

    # Two entities that collide on the id triple but differ by entity_key.
    sink.upsert(
        spark.createDataFrame([(1, 500, 7, 0, 10, key_47)], _NEW_SCHEMA), uri, _ENTITY_MERGE_KEYS
    )
    sink.upsert(
        spark.createDataFrame([(1, 500, 7, 0, 10, key_88)], _NEW_SCHEMA), uri, _ENTITY_MERGE_KEYS
    )

    result = spark.table(uri)
    assert result.count() == 2  # both entities preserved, not overwritten
    assert {r.entity_key for r in result.collect()} == {key_47, key_88}

    # Re-upserting entity 47 with a changed window updates in place (still 2 rows).
    sink.upsert(
        spark.createDataFrame([(1, 500, 7, 0, 99, key_47)], _NEW_SCHEMA), uri, _ENTITY_MERGE_KEYS
    )
    result = spark.table(uri)
    assert result.count() == 2
    by_key = {r.entity_key: r for r in result.collect()}
    assert by_key[key_47].end_ts == 99  # updated
    assert by_key[key_88].end_ts == 10  # untouched
    spark.sql(f"DROP TABLE IF EXISTS {uri}")


def test_upsert_null_entity_key_is_idempotent(spark):
    # BasicEvent rows carry a NULL entity_key and share the event-fact table.
    # With entity_key in the merge key, the condition must be NULL-safe (`<=>`)
    # so re-upserting the same row updates rather than duplicating it.
    uri = "spark_catalog.gold.upsert_null_entity_key_test"
    spark.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.gold")
    spark.sql(f"DROP TABLE IF EXISTS {uri}")

    sink = UnityCatalogSink(
        UnitySinkConfig(catalog_name="spark_catalog", schema_name="gold", table_prefix="")
    )
    row = [(1, 600, 7, 0, 10, None)]
    sink.upsert(spark.createDataFrame(row, _NEW_SCHEMA), uri, _ENTITY_MERGE_KEYS)
    sink.upsert(spark.createDataFrame(row, _NEW_SCHEMA), uri, _ENTITY_MERGE_KEYS)

    assert spark.table(uri).count() == 1  # NULL <=> NULL matched → updated, not duplicated
    spark.sql(f"DROP TABLE IF EXISTS {uri}")
