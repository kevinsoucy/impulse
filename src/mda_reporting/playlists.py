"""Conversions involving the versioned `playlist_items` table.

A playlist is a named, versioned collection of event windows. The original
producer is the TSAL search step, which writes one playlist row per matched
`event_instance_fact` row; future producers (manual curation, ML re-ranking)
reuse the same shape.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def event_fact_to_playlist_items(
    events_df: DataFrame,
    playlist_id: str,
    playlist_version: int = 1,
    event_name: str = "pedestrian_high_speed_proximity",
    now_utc: datetime | None = None,
) -> DataFrame:
    """Project rows from `event_instance_fact` into the `playlist_items` shape.

    `event_id` is derived as a stable SHA-256 over (container_id, start_ts, end_ts)
    so re-running the search produces the same IDs and `playlist_items.event_id`
    is round-trip-stable across runs.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    return (
        events_df
        .withColumn(
            "playlist_event_id",
            F.expr(
                "substr(sha2(concat_ws('-', cast(container_id as string), "
                "cast(start_ts as string), cast(end_ts as string)), 256), 1, 32)"
            ),
        )
        .select(
            F.col("container_id"),
            F.col("playlist_event_id").alias("event_id"),
            F.lit(event_name).alias("event_name"),
            F.col("start_ts").cast("long").alias("start_ts"),
            F.col("end_ts").cast("long").alias("end_ts"),
            F.lit(playlist_id).alias("playlist_id"),
            F.lit(playlist_version).cast("int").alias("playlist_version"),
            F.lit(now_utc).alias("created_at"),
        )
    )
