"""
twitter_stream.py
-----------------

Implements real-time Twitter streaming.

Design goals:
- Resilient ingestion
- 1500 tweets/sec capability
- Compatible with Apache Flume

Uses Tweepy v4 streaming interface.
"""

import json
import logging
from typing import List
import tweepy

from .config import TwitterConfig

logger = logging.getLogger(__name__)


class TwitterStreamer(tweepy.StreamingClient):
    """
    Custom Twitter streaming client.

    Each incoming tweet is serialized and forwarded to Flume.
    """

    def __init__(self, bearer_token: str, flume_sink):
        super().__init__(bearer_token)
        self.flume_sink = flume_sink

    def on_tweet(self, tweet):
        """Handle incoming tweet."""
        try:
            payload = {
                "id": tweet.id,
                "text": tweet.text,
                "created_at": str(tweet.created_at),
            }
            self.flume_sink.send(json.dumps(payload))
        except Exception as exc:
            logger.error("Tweet processing failed: %s", exc)


def start_stream(track_terms: List[str], flume_sink):
    """
    Start Twitter streaming.

    Parameters
    ----------
    track_terms : List[str]
        Keywords to track (AI, ML, Hadoop, etc.)
    flume_sink : object
        Flume sink client
    """

    config = TwitterConfig()
    streamer = TwitterStreamer(config.API_KEY, flume_sink)

    rules = [tweepy.StreamRule(term) for term in track_terms]
    streamer.add_rules(rules)

    streamer.filter(threaded=True)
