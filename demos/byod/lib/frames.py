"""Frame-lookup helper for the BYOD ADAS demo.

``frame_nearest_to`` is a standalone Spark query over ``perception_channels``
with no engine dependency — it just picks the media frame closest to a timestamp,
used when rendering a visualization for an event.
"""

from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F


def frame_nearest_to(
    spark: SparkSession,
    perception_channels_table: str,
    container_id: int,
    channel_id: int,
    target_ts: int,
) -> Row:
    """Return the ``perception_channels`` row whose ``timestamp`` is closest to
    ``target_ts`` for the given ``(container_id, channel_id)``.

    Used to pick the camera frame or LiDAR scan nearest to an event's midpoint
    when rendering a visualization.
    """
    return (
        spark.read.table(perception_channels_table)
        .filter((F.col("container_id") == container_id) & (F.col("channel_id") == channel_id))
        .withColumn("dt", F.abs(F.col("timestamp") - target_ts))
        .orderBy("dt")
        .first()
    )
