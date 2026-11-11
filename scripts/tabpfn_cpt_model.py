"""Model operations frozen from the independently verified eight-case TabPFN run."""
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import time
import warnings
import numpy as np
from tabpfn_cpt_protocol import query_grid, training_arrays


def initialize_torch():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("This fixed baseline requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    return torch


def estimator_options(torch):
    return dict(n_estimators=8, device="cuda", random_state=0, inference_precision=torch.float32,
                memory_saving_mode=False, fit_mode="fit_preprocessors", ignore_pretraining_limits=True,
                n_preprocessing_jobs=1, softmax_temperature=.9, average_before_softmax=False)


def make_regressor(model_path, torch):
    from tabpfn import TabPFNRegressor
    from tabpfn.base import RegressorModelSpecs
    from tabpfn.model_loading import load_model
    model, criterion, architecture, inference = load_model(
        path=Path(model_path), cache_trainset_representation=False, estimator_type="regressor")
    spec = RegressorModelSpecs(model=model, architecture_config=architecture,
                               inference_config=inference, norm_criterion=criterion)
    return TabPFNRegressor(model_path=spec, **estimator_options(torch)), model


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def predict_source(regressor, source, mask, torch):
    # Evaluation targets are deliberately absent from this function's arguments.
    X, y = training_arrays(source, mask)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        regressor.fit(X, y)
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        prediction = regressor.predict(query_grid(), output_type="median")
        torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - tick
    prediction = np.asarray(prediction, dtype=np.float32).reshape(32, 512)
    assert np.isfinite(prediction).all()
    assert regressor.n_estimators_ == 8
    assert regressor.n_features_in_ == 2
    diagnostics = dict(n_observations=len(y), resolved_n_estimators=int(regressor.n_estimators_),
        inference_config=json.loads(json.dumps(dataclasses.asdict(regressor.inference_config_), default=str)),
        warnings=[dict(category=type(w.message).__name__, message=str(w.message)) for w in caught])
    return prediction, diagnostics, dict(fit_seconds=fit_seconds, predict_seconds=predict_seconds)
