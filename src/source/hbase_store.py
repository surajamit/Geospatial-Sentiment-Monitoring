"""
hbase_store.py
--------------

Stores classified tweets into HBase for fast lookup.

Why HBase:
- low latency reads
- scalable NoSQL layer
- complements HDFS cold storage
"""

import happybase
from typing import Dict


class HBaseClient:
    """Minimal HBase interface."""

    def __init__(self, host: str = "localhost"):
        self.connection = happybase.Connection(host)

    def store_tweet(self, table_name: str, row_key: str, data: Dict):
        """Insert tweet record."""
        table = self.connection.table(table_name)
        table.put(row_key.encode(), {
            b"meta:text": data["text"].encode(),
            b"meta:label": str(data["label"]).encode(),
        })
