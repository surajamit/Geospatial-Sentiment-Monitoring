"""
KDE Spatial Density Estimator
=============================

Implements non-parametric spatial intensity estimation.

Mathematical Formulation
------------------------
Given geo-points x_i:

    f(x) = (1 / (n h^2)) * Σ K( (x - x_i) / h )

where:
    h = bandwidth
    K = Gaussian kernel

Used to generate smooth technology heatmaps.
"""

import numpy as np
from sklearn.neighbors import KernelDensity


def compute_kde_density(coords: np.ndarray,
                        bandwidth: float = 0.5,
                        grid_size: int = 100):
    """
    Estimate spatial density via KDE.

    Parameters
    ----------
    coords : array (n,2)
        Latitude/longitude pairs
    bandwidth : float
        KDE bandwidth (h)
    grid_size : int
        Resolution of heatmap

    Returns
    -------
    X, Y, Z : meshgrid + density
    """

    kde = KernelDensity(
        bandwidth=bandwidth,
        kernel="gaussian"
    ).fit(coords)

    # grid bounds
    lat_min, lon_min = coords.min(axis=0)
    lat_max, lon_max = coords.max(axis=0)

    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)

    X, Y = np.meshgrid(lat_grid, lon_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    log_density = kde.score_samples(grid_points)
    Z = np.exp(log_density).reshape(grid_size, grid_size)

    return X, Y, Z
