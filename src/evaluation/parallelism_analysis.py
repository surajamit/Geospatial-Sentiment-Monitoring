"""
Parallelism and Scalability Analysis Module
==========================================

Provides quantitative proof of Spark parallel execution.

Mathematical Formulation
------------------------

Speedup:
    S(p) = T(1) / T(p)

Efficiency:
    E(p) = S(p) / p

Where:
    p = number of partitions (proxy for parallel workers)
    T(p) = execution time with p partitions

Throughput scaling:
    Θ(p) = N_processed / T(p)

Used to validate near-linear scalability claims.
"""

import time
import numpy as np
from typing import Dict, List


# --------------------------------------------------
# Speedup
# --------------------------------------------------

def compute_speedup(t1: float, tp: float) -> float:
    """S(p) = T(1) / T(p)"""
    if tp == 0:
        return 0.0
    return t1 / tp


# --------------------------------------------------
# Efficiency
# --------------------------------------------------

def compute_efficiency(speedup: float, partitions: int) -> float:
    """E(p) = S(p) / p"""
    if partitions == 0:
        return 0.0
    return speedup / partitions


# --------------------------------------------------
# Throughput
# --------------------------------------------------

def compute_throughput(records: int, exec_time: float) -> float:
    """Θ(p) = N / T(p)"""
    if exec_time == 0:
        return 0.0
    return records / exec_time


# --------------------------------------------------
# Benchmark Runner
# --------------------------------------------------

def benchmark_partitions(
    spark,
    data_size: int,
    partition_list: List[int]
) -> List[Dict]:
    """
    Run controlled scalability benchmark.

    Parameters
    ----------
    spark : SparkSession
    data_size : int
        Number of synthetic tweets
    partition_list : list
        Partition configurations

    Returns
    -------
    list of dict
        Benchmark results
    """

    results = []

    # synthetic workload
    base_rdd = spark.sparkContext.parallelize(
        range(data_size),
        numSlices=1
    )

    # baseline time T(1)
    start = time.time()
    base_rdd.map(lambda x: x * x).count()
    t1 = time.time() - start

    for p in partition_list:

        rdd = spark.sparkContext.parallelize(
            range(data_size),
            numSlices=p
        )

        start = time.time()
        rdd.map(lambda x: x * x).count()
        tp = time.time() - start

        speedup = compute_speedup(t1, tp)
        efficiency = compute_efficiency(speedup, p)
        throughput = compute_throughput(data_size, tp)

        results.append({
            "partitions": p,
            "time_sec": round(tp, 4),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 3),
            "throughput_rec_per_sec": round(throughput, 2),
        })

    return results
