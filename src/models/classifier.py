"""
classifier.py
-------------

Random Forest classifier (Proposed Model).

parameters:
- 200 trees
- max depth 20
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

from .config import ModelConfig

MODEL_PATH = "data/processed/rf_model.joblib"


def build_model() -> RandomForestClassifier:
    """Create RF model with fixed hyperparameters."""
    return RandomForestClassifier(
        n_estimators=ModelConfig.N_ESTIMATORS,
        max_depth=ModelConfig.MAX_DEPTH,
        random_state=ModelConfig.RANDOM_STATE,
        n_jobs=-1,
    )


def save_model(model):
    joblib.dump(model, MODEL_PATH)


def load_rf_model():
    return joblib.load(MODEL_PATH)
