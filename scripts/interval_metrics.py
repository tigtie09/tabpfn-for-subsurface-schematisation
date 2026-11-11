"""Pointwise interval definitions used in the synthetic experiments."""

from statistics import NormalDist
import numpy as np


def normal_bounds(mean, std, levels):
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std >= 0).all()
    critical = np.array([NormalDist().inv_cdf((1 + p) / 2) for p in levels])
    radius = critical[:, None, None] * std
    return np.stack((mean - radius, mean + radius), axis=1)


def score_intervals(truth, observed, bounds, levels):
    assert bounds.shape == (len(levels), 2, 32, 512)
    assert np.isfinite(bounds).all() and (bounds[:, 0] <= bounds[:, 1]).all()
    assert (np.diff(bounds[:, 0], axis=0) <= 1e-6).all()
    assert (np.diff(bounds[:, 1], axis=0) >= -1e-6).all()
    rows = []
    for scope, mask in (
        ("unobserved_pixels", ~observed),
        ("all_pixels", np.ones_like(observed)),
    ):
        y = truth[mask].astype(np.float64)
        for level, interval in zip(levels, bounds):
            low, high = interval[:, mask].astype(np.float64)
            width = high - low
            score = width + 2 / (1 - level) * (
                np.maximum(low - y, 0) + np.maximum(y - high, 0)
            )
            rows.append(
                dict(
                    scope=scope,
                    level=level,
                    n_pixels=len(y),
                    coverage=float(np.mean((low <= y) & (y <= high))),
                    below_rate=float(np.mean(y < low)),
                    above_rate=float(np.mean(y > high)),
                    width_mean=float(width.mean()),
                    width_median=float(np.median(width)),
                    interval_score=float(score.mean()),
                )
            )
    return rows
