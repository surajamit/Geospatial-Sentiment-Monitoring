"""
train_rf.py
Random Forest sentiment classifier.

Manuscript parameters:
    trees = 200
    depth = 20
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from src.config import RFConfig


def build_model(cfg: RFConfig) -> Pipeline:
    """Construct RF pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("rf", RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=-1
        ))
    ])


def train_model(df: pd.DataFrame) -> Pipeline:
    """
    Train sentiment classifier.

    Expected columns:
        - text
        - label
    """
    cfg = RFConfig()
    model = build_model(cfg)

    X = df["text"]
    y = df["label"]

    model.fit(X, y)
    return model
