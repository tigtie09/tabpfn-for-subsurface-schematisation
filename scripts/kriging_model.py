"""Kriging fitted with the released SchemaGAN settings."""

import json
import time
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def fit_predict(source, mask, seed):
    source, mask = np.asarray(source), np.asarray(mask)
    if (
        source.ndim != 2
        or mask.shape != source.shape
        or mask.dtype != np.bool_
        or not mask.any()
    ):
        raise ValueError("Need a 2D source and a nonempty boolean observation mask")
    if not np.isfinite(source[mask]).all():
        raise ValueError("Input observations must be finite")
    zz, xx = np.nonzero(mask)
    points = np.column_stack((xx, zz)).astype(np.float64)
    # The original wrapper passes a Python list, which sklearn converts to float64.
    values = source[mask].astype(np.float64)
    gx, gy = np.meshgrid(
        np.linspace(0, source.shape[1] - 1, source.shape[1]),
        np.linspace(0, source.shape[0] - 1, source.shape[0]),
    )
    queries = np.column_stack((gx.ravel(), gy.ravel()))
    model = GaussianProcessRegressor(
        kernel=RBF(length_scale=(50.0, 0.5), length_scale_bounds=(1e-5, 1e5))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5)),
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=int(seed),
        copy_X_train=True,
    )
    tick = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(points, values)
    fit_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    raw_prediction = model.predict(queries).reshape(source.shape)
    predict_seconds = time.perf_counter() - tick
    if not np.isfinite(raw_prediction).all():
        raise ValueError("Nonfinite GP prediction")
    prediction = raw_prediction.astype(np.float32)
    parameters = dict(
        random_state=int(seed),
        n_observations=len(values),
        length_scale_x=float(model.kernel_.k1.length_scale[0]),
        length_scale_z=float(model.kernel_.k1.length_scale[1]),
        noise_level=float(model.kernel_.k2.noise_level),
        alpha=float(model.alpha),
        y_mean=float(model._y_train_mean),
        y_std=float(model._y_train_std),
        log_marginal_likelihood=float(model.log_marginal_likelihood_value_),
        warning_count=len(caught),
        warnings=json.dumps(
            [
                dict(category=type(w.message).__name__, message=str(w.message))
                for w in caught
            ]
        ),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )
    return prediction, model.alpha_.copy(), parameters
