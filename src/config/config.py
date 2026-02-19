"""
Author: Amit Pimpalkar
Python: 3.9
Spark: 3.x
"""

"""
config.py
---------

Central configuration for FUSION-SPARK Geo-Sentiment system.

Contains:
- Twitter API placeholders
- Spark parameters
- RF hyperparameters
- Cluster configuration

"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TwitterConfig:
    """Twitter API credentials (xxxxxxxxxxxxxxxxxxxxxxxxxx)."""
    API_KEY: str = os.getenv("TWITTER_API_KEY", "API_KEY")
    API_SECRET: str = os.getenv("TWITTER_API_SECRET", "API_SECRET")
    ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN", "ACCESS_TOKEN")
    ACCESS_SECRET: str = os.getenv("TWITTER_ACCESS_SECRET", "ACCESS_SECRET")


@dataclass(frozen=True)
class SparkConfig:
    """Spark streaming configuration aligned with manuscript."""
    APP_NAME: str = "FUSION-SPARK-GEO-SENTIMENT"
    MASTER: str = "local[*]"
    MICRO_BATCH: int = 5  # seconds
    NUM_CORES_CLUSTER: int = 32
    CLUSTER_MEMORY_GB: int = 128


@dataclass(frozen=True)
class ModelConfig:
    """Random Forest configuration."""
    N_ESTIMATORS: int = 200
    MAX_DEPTH: int = 20
    RANDOM_STATE: int = 42


@dataclass(frozen=True)
class HDFSConfig:
    """HDFS paths."""
    HDFS_URI: str = "hdfs://192.168.1.165:9000"
    FLUME_DIR: str = "/user/flume/ml/"
