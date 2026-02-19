"""
flume_channel_sink.py
---------------------

Implements the conceptual Flume flow:

Source → Channel → Sink

This architecture supports fault-tolerant buffering before HDFS persistence.
"""

import queue
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemoryChannel:
    """
    Simulated Flume memory channel.

    Acts as decoupling buffer between source and sink.
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = queue.Queue(maxsize=capacity)

    def put(self, event: Any):
        """Insert event into channel."""
        self.buffer.put(event)

    def get(self) -> Any:
        """Retrieve event from channel."""
        return self.buffer.get()


class HDFSSink:
    """
    Simulated Flume sink writing to HDFS.

    In production this maps to:
    Hadoop fs -put
    """

    def __init__(self, hdfs_path: str):
        self.hdfs_path = hdfs_path

    def write(self, event: str):
        """Persist event to HDFS."""
        # placeholder for real HDFS client
        with open("data/raw/flume_stream.log", "a") as f:
            f.write(event + "\n")
