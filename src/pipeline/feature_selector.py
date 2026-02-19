"""
Feature Selection Module
------------------------
Selects the five key attributes.
"""

from pyspark.sql import DataFrame


SELECTED_FIELDS = [
    "text",
    "city",
    "location",
    "latitude",
    "longitude",
]


def select_core_features(df: DataFrame) -> DataFrame:
    """
    Extract key analytical fields.

    Returns structured DataFrame ready for ML.
    """
    return df.select(*SELECTED_FIELDS)
