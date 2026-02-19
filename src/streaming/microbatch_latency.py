"""
Micro-Batch Latency Profiler
===========================

Measures Spark processing latency per batch.

Latency:
    L_k = t_end - t_start
"""

import time


class MicroBatchProfiler:

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def latency(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
