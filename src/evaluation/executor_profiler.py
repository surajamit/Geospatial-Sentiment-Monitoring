"""
Executor-Level Resource Profiler
================================

Provides runtime monitoring of Spark executor CPU and memory usage.

Metrics captured
----------------
1. Executor memory utilization
2. Executor task distribution
3. Parallel workload balance
4. Resource efficiency indicators

Works with Spark 3.x via statusTracker and REST metrics.
"""

import psutil
import time
from typing import Dict, List


# --------------------------------------------------
# Driver Resource Snapshot (local fallback)
# --------------------------------------------------

def get_driver_resource_usage() -> Dict:
    """
    Capture CPU and memory usage of the driver node.

    Useful when executor metrics are partially available.
    """

    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": round(memory.used / (1024**3), 2),
        "memory_total_gb": round(memory.total / (1024**3), 2),
    }


# --------------------------------------------------
# Executor Status via Spark
# --------------------------------------------------

def get_executor_info(spark) -> List[Dict]:
    """
    Extract executor-level statistics from Spark.

    Returns
    -------
    list of dict
        Executor metrics
    """

    sc = spark.sparkContext
    status_list = sc.statusTracker().getExecutorInfos()

    executors = []

    for info in status_list:
        executors.append({
            "executor_id": info.executorId(),
            "host": info.hostPort(),
            "total_cores": info.totalCores(),
            "max_memory_mb": round(info.maxMemory() / (1024 * 1024), 2),
        })

    return executors


# --------------------------------------------------
# Parallelism Balance Metric
# --------------------------------------------------

def compute_parallelism_balance(task_counts: List[int]) -> float:
    """
    Measure load balance across executors.

    Balance ≈ 1.0 means well balanced.

    Formula:
        Balance = min(tasks) / max(tasks)
    """

    if not task_counts or max(task_counts) == 0:
        return 0.0

    return min(task_counts) / max(task_counts)


# --------------------------------------------------
# Periodic Monitor
# --------------------------------------------------

def monitor_resources(interval_sec: int = 5, duration_sec: int = 60):
    """
    Periodically log driver resource usage.
    """

    snapshots = []

    steps = duration_sec // interval_sec

    for _ in range(steps):
        usage = get_driver_resource_usage()
        usage["timestamp"] = time.time()
        snapshots.append(usage)
        time.sleep(interval_sec)

    return snapshots
