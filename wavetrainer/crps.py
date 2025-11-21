"""A function for constructing the continuous ranked probability score."""

# pylint: disable=invalid-name

import numpy as np
import xarray as xr
from scores.probability import crps_cdf  # type: ignore


def _cdf_from_quantiles(
    qs: np.ndarray, q_vals: np.ndarray, z_grid: np.ndarray
) -> np.ndarray:
    qs = np.asarray(qs)
    q_vals = np.asarray(q_vals)
    z_grid = np.asarray(z_grid)

    F = np.zeros_like(z_grid, dtype=float)

    # Below minimum quantile: prob ~ 0
    F[z_grid <= q_vals[0]] = 0.0
    # Above maximum quantile: prob ~ 1
    F[z_grid >= q_vals[-1]] = 1.0

    # Interior: invert quantile function by linear interpolation
    mask_mid = (z_grid > q_vals[0]) & (z_grid < q_vals[-1])
    z_mid = z_grid[mask_mid]

    # For each z, find the enclosing quantile interval [q_vals[i], q_vals[i+1]]
    idx = np.searchsorted(q_vals, z_mid) - 1
    idx = np.clip(idx, 0, len(q_vals) - 2)

    q0 = q_vals[idx]
    q1 = q_vals[idx + 1]
    a0 = qs[idx]
    a1 = qs[idx + 1]

    # Linear interpolation in value-space → CDF-space
    t = (z_mid - q0) / (q1 - q0)
    F_mid = a0 + t * (a1 - a0)

    F[mask_mid] = F_mid
    return F


def _batch_crps_scores(
    y_true_array: np.ndarray,
    qs: np.ndarray,
    q_vals_batch: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    N = len(y_true_array)
    scores_out = np.empty(N)

    for i in range(N):
        F_grid = _cdf_from_quantiles(qs, q_vals_batch[i], z_grid)

        fcst_cdf_da = xr.DataArray(
            data=F_grid,
            coords={"z": z_grid},
            dims=["z"],
            name="forecast_cdf",
        )
        obs_da = xr.DataArray(float(y_true_array[i]), name="obs")

        crps_result = crps_cdf(fcst_cdf_da, obs_da, threshold_dim="z")
        scores_out[i] = float(crps_result.total)

    return scores_out


def crps(y: np.ndarray, y_q: np.ndarray, quantiles: np.ndarray) -> float:
    """Compute the mean CRPS for the quantile regression predictions."""
    z_grid = np.linspace(-5, 5, 201)  # e.g. z from -5 to +5
    crps_scores = _batch_crps_scores(y, quantiles, y_q, z_grid)
    return crps_scores.mean()
