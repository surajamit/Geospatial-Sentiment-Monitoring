"""
Flume Throughput Monitor
=======================

Validates ingestion rate from Flume channel.

Throughput formula:
    T = N_events / Δt
"""

import time


class FlumeRateMonitor:

    def __init__(self, window_sec: int = 10):
        self.window_sec = window_sec
        self.counter = 0
        self.start_time = time.time()

    def record_event(self, n: int = 1):
        """Call for each ingested tweet."""
        self.counter += n

    def compute_rate(self) -> float:
        """Compute tweets/sec."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.counter / elapsed

    def reset(self):
        self.counter = 0
        self.start_time = time.time()
