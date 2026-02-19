"""
Micro-Batch Streaming Engine
=============================

Implements the formal streaming model.

Mathematical Model
------------------
Incoming Twitter stream:

    S = {x_i, t_i}

Micro-batch formation:

    B_k = { x_i | kΔt ≤ t_i < (k+1)Δt }

Ingestion throughput:

    T = |B_k| / Δt   (tweets per second)

Processing latency:

    L = t_process_end - t_batch_start

This module provides reusable utilities for monitoring real-time Spark Streaming performance.

Compatible with Spark 3.x Structured Streaming.
"""

import time
from dataclasses import dataclass
from typing import Dict


# --------------------------------------------------
# Throughput Calculator
# --------------------------------------------------

def compute_throughput(batch_size: int, batch_interval: float) -> float:
    """
    Compute ingestion throughput.

    T = |B_k| / Δt

    Parameters
    ----------
    batch_size : int
        Number of tweets in micro-batch
    batch_interval : float
        Micro-batch duration in seconds

    Returns
    -------
    float
        Tweets per second
    """
    if batch_interval == 0:
        return 0.0
    return batch_size / batch_interval


# --------------------------------------------------
# Latency Monitor
# --------------------------------------------------

@dataclass
class LatencyTracker:
    """Track micro-batch latency."""

    start_time: float = 0.0
    end_time: float = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def latency(self) -> float:
        """Return processing latency in seconds."""
        return self.end_time - self.start_time


# --------------------------------------------------
# Streaming Metrics Logger
# --------------------------------------------------

def log_streaming_metrics(batch_size: int,
                          batch_interval: float,
                          latency: float) -> Dict:
    """
    Generate metrics dictionary for monitoring.
    """

    throughput = compute_throughput(batch_size, batch_interval)

    return {
        "batch_size": batch_size,
        "batch_interval_sec": batch_interval,
        "throughput_tweets_per_sec": round(throughput, 2),
        "processing_latency_sec": round(latency, 3),
    }
