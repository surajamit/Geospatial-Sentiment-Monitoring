"""
spark_stream_processor.py
High-performance Spark Streaming pipeline.

Parallelism mechanism:
    - RDD partitioning
    - Distributed map transformations
"""

from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from src.config import SparkConfig
from src.preprocessing import preprocess_record


def create_spark_session(cfg: SparkConfig) -> SparkSession:
    """Initialize Spark session."""
    return (
        SparkSession.builder
        .appName(cfg.app_name)
        .master(cfg.master)
        .getOrCreate()
    )


def process_rdd(rdd):
    """RDD transformation logic."""
    if rdd.isEmpty():
        return

    processed = rdd.map(lambda x: preprocess_record(x))
    print(f"Processed partitions: {processed.getNumPartitions()}")


def start_streaming(socket_host="localhost", socket_port=9999):
    """
    Start Spark Streaming context.

    Micro-batch = 5 seconds (as in manuscript).
    """
    cfg = SparkConfig()

    spark = create_spark_session(cfg)
    ssc = StreamingContext(spark.sparkContext, cfg.batch_interval)

    stream = ssc.socketTextStream(socket_host, socket_port)

    parsed = stream.map(lambda x: {"text": x})
    parsed.foreachRDD(process_rdd)

    ssc.start()
    ssc.awaitTermination()
