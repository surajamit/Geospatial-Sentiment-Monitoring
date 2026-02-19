"""
spark_streaming.py
------------------

Implements Spark Streaming using RDD transformations

Pipeline:
Flume → Spark Streaming → RDD → Preprocessing → RF → HDFS
"""

import logging
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from .config import SparkConfig, HDFSConfig
from .preprocessing import preprocess_tweet
from .classifier import load_rf_model

logger = logging.getLogger(__name__)


def create_streaming_context():
    """Create Spark Streaming context."""

    sc = SparkContext(appName=SparkConfig.APP_NAME)
    ssc = StreamingContext(sc, SparkConfig.MICRO_BATCH)

    return sc, ssc


def process_rdd(rdd):
    """
    Core RDD processing logic.

    Each Spark agent executes:
    - cleaning
    - tokenization
    - classification
    - write to HDFS
    """

    if rdd.isEmpty():
        return

    model = load_rf_model()

    processed = (
        rdd.map(lambda x: preprocess_tweet(x))
           .map(lambda x: (x, model.predict([x])[0]))
    )

    processed.saveAsTextFile(
        f"{HDFSConfig.HDFS_URI}{HDFSConfig.FLUME_DIR}/classified"
    )


def start_streaming(ssc):
    """Attach Flume stream and start processing."""

    # Placeholder socket simulating Flume
    stream = ssc.socketTextStream("localhost", 9999)

    stream.foreachRDD(process_rdd)

    ssc.start()
    ssc.awaitTermination()
