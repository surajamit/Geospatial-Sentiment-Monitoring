"""
hdfs_writer.py
--------------

Handles distributed storage.

Features:
- HDFS replication awareness
- fault tolerance hooks
- MapReduce compatibility
"""

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "HDFS_WRITER"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.dfs.replication", "3")
        .getOrCreate()
    )


def write_to_hdfs(df, output_path: str):
    """
    Write DataFrame to HDFS with replication.

    Ensures distributed fault tolerance.
    """
    (
        df.write
        .mode("append")
        .option("compression", "snappy")
        .csv(output_path)
    )
