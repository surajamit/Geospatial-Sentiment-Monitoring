"""
heatmap.py
----------

Kernel Density Estimation spatial visualization.

Generates heatmaps for:
- AI
- ML
- Python
- CC
- Hadoop
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_kde_heatmap(df: pd.DataFrame, tech_name: str):
    """
    Generate spatial KDE heatmap.

    Parameters
    ----------
    df : DataFrame
        Must contain latitude and longitude.
    tech_name : str
        Technology label.
    """

    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        x=df["longitude"],
        y=df["latitude"],
        fill=True,
        cmap="Reds",
        thresh=0.05,
    )
    plt.title(f"KDE Density — {tech_name}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
