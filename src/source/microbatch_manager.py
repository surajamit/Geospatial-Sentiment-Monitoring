"""
microbatch_manager.py
---------------------

Implements formal micro-batch tracking B_k.
Provides throughput and latency instrumentation.
"""

import time
from dataclasses import dataclass


@dataclass
class BatchStats:
    batch_id: int
    size: int
    duration: float
    throughput: float


class MicroBatchMonitor:
    """
    Tracks Spark Streaming performance metrics.
    """

    def __init__(self, delta_t: float):
        self.delta_t = delta_t
        self.batch_id = 0

    def evaluate_batch(self, batch_size: int, start_time: float) -> BatchStats:
        """Compute throughput T = |B_k| / Δt."""
        duration = time.time() - start_time
        throughput = batch_size / self.delta_t

        stats = BatchStats(
            batch_id=self.batch_id,
            size=batch_size,
            duration=duration,
            throughput=throughput,
        )

        self.batch_id += 1
        return stats
