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
# 2.0-shaped rows — adds the nullable entity_key.
_NEW_SCHEMA = T.StructType(
    [*_LEGACY_SCHEMA.fields, T.StructField("entity_key", T.StringType(), True)]
)

_MERGE_KEYS = ["container_id", "event_id", "event_instance_id"]
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
