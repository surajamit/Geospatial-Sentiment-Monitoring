"""
Experiment Configuration Logger
==============================

Records environment details.

Captures:
- Spark version
- Python version
- scikit-learn version
- cluster resources
- micro-batch interval
- RF parameters

"""

import platform
import sklearn
import pyspark
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    spark_version: str
    python_version: str
    sklearn_version: str
    total_cores: int
    total_memory_gb: int
    micro_batch_sec: int
    rf_trees: int
    rf_depth: int
    flume_target_rate: int


def capture_experiment_config(spark) -> dict:
    """Capture full runtime configuration."""

    sc = spark.sparkContext

    config = ExperimentConfig(
        spark_version=pyspark.__version__,
        python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,
        total_cores=sc.defaultParallelism,
        total_memory_gb=128,  # as per cluster spec
        micro_batch_sec=5,
        rf_trees=200,
        rf_depth=20,
        flume_target_rate=1500,
    )

    return asdict(config)
