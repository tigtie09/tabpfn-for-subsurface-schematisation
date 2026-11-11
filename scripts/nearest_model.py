"""Pixel-distance nearest-neighbour prediction; no field-data dependencies."""

import numpy as np
from scipy.spatial import cKDTree


def nearest_grid(values, mask):
    """Return IC and donor flat indices; missing pixels never enter the tree."""
    values, mask = np.asarray(values), np.asarray(mask)
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype != np.bool_:
        raise ValueError("Expected a 2D image and a matching boolean observation mask")
    if not mask.any() or not np.isfinite(values[mask]).all():
        raise ValueError(
            "Need at least one finite observation and no invalid observations"
        )
    points = np.column_stack(np.nonzero(mask))
    queries = np.indices(values.shape).reshape(2, -1).T
    _, indices = cKDTree(points).query(queries, k=1, eps=0, p=2, workers=1)
    donors = np.flatnonzero(mask)[indices].reshape(values.shape)
    return values.ravel()[donors], donors
