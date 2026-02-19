"""
preprocessing.py
----------------

Tweet preprocessing.

Steps:
- text cleaning
- normalization
- tokenization
- stopword removal
"""

import re
from typing import List

STOPWORDS = {"the", "is", "at", "on", "in", "and"}

def clean_tweet(text: str) -> str:
    """Remove noise from tweet text."""
    """Remove URLs, mentions, hashtags."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return [t for t in text.split() if t not in STOPWORDS]


def preprocess_tweet(text: str) -> str:
    """Full preprocessing pipeline."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return " ".join(tokens)


"""
Feature engineering utilities.

Implements:
    x_i -> f(x_i)

Used in Spark RDD pipeline.
"""

import re
import string
from typing import List

def preprocess_record(record: dict) -> dict:
    """
    Apply full preprocessing pipeline.

    Parameters
    ----------
    record : dict
        Raw tweet record

    Returns
    -------
    dict
        Processed tweet
    """
    cleaned = clean_tweet(record.get("text", ""))
    tokens = tokenize(cleaned)

    return {
        **record,
        "clean_text": cleaned,
        "tokens": tokens,
    }
