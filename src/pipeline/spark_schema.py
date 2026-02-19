"""
Spark Schema Definition for Twitter Data
----------------------------------------
Implements structured schema.
"""

from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, LongType
)


def get_twitter_schema() -> StructType:
    """Full tweet + user schema."""

    return StructType([
        StructField("tweet_id", LongType(), True),
        StructField("text", StringType(), True),
        StructField("city", StringType(), True),
        StructField("location", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("user_id", LongType(), True),
        StructField("username", StringType(), True),
        StructField("followers", LongType(), True),
        StructField("created_at", StringType(), True),
    ])
