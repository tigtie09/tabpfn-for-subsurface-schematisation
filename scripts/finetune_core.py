"""CPT fine-tuning with context-only preprocessing and aligned target transforms."""
from __future__ import annotations
import copy
import dataclasses
import hashlib
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.base import RegressorModelSpecs
from tabpfn.finetuning.data_util import DatasetCollectionWithPreprocessing, RegressorDatasetConfig, meta_dataset_collator
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss
from tabpfn.model_loading import load_model
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn_cpt_model import estimator_options, parameter_digest
from tabpfn_cpt_protocol import query_grid, training_arrays

GRID = query_grid()
PERFORMANCE = PerformanceOptions(force_recompute_layer=True, use_chunkwise_inference=False)


def initialize(device, seed=0):
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type(device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the full experiment")
        torch.cuda.set_per_process_memory_fraction(.40)


def estimators(checkpoint, device):
    model, criterion, arch, inference = load_model(path=Path(checkpoint), cache_trainset_representation=False, estimator_type="regressor")
    def spec():
        return RegressorModelSpecs(model=model, architecture_config=arch, inference_config=copy.deepcopy(inference), norm_criterion=copy.deepcopy(criterion))
    options = dict(estimator_options(torch), device=device, show_progress_bar=False)
    train = TabPFNRegressor(model_path=spec(), **dict(options, n_estimators=1, fit_mode="batched"))
    infer = TabPFNRegressor(model_path=spec(), **options)
    train._initialize_model_variables()
    infer._initialize_model_variables()
    assert train.models_[0] is infer.models_[0] is model
    model.to(device)
    families = [(copy.deepcopy(x), y) for x in train.inference_config_.PREPROCESS_TRANSFORMS
                for y in train.inference_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS]
    assert len(families) == 4 and {y for _, y in families} == {None, "safepower"}
    return train, infer, model, families


def make_batch(estimator, source, truth, family, seed, n_query=2048):
    mask = source != 0
    context_x, context_y = training_arrays(source, mask)
    np.testing.assert_array_equal(context_y, truth[mask])
    rng = np.random.default_rng(seed)
    indices = np.flatnonzero(~mask.ravel())
    query = rng.choice(indices, min(n_query, len(indices)), replace=False)
    query_x, query_y = GRID[query], truth.ravel()[query].astype(np.float32)
    assert not np.any(mask.ravel()[query])
    x_transform, y_transform = family
    estimator.inference_config_.PREPROCESS_TRANSFORMS = [copy.deepcopy(x_transform)]
    estimator.inference_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS = [y_transform]
    # Initialize feature detection and fitted transforms from observed values only.
    configs, X, y, criterion = estimator._initialize_dataset_preprocessing(context_x, context_y, random_state=seed)
    assert not estimator.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL)
    def split(observed_x, observed_y, stratify=None):
        return observed_x, query_x, observed_y, query_y
    data_config = RegressorDatasetConfig(config=configs, X_raw=X, y_raw=y, cat_ix=[], znorm_space_bardist_=criterion)
    dataset = DatasetCollectionWithPreprocessing(split_fn=split, random_state=seed, dataset_config_collection=[data_config], stratify=False, n_preprocessing_jobs=1)
    item = dataset[0]
    batch = meta_dataset_collator([item])
    transform = batch.configs[0][0].target_transform
    # The library's batch target is only z-normalized. Apply the same fitted
    # nonlinear transform as the context before using raw model bar logits.
    target = batch.y_query.cpu().numpy().reshape(-1, 1)
    expected_context = ((context_y - context_y.mean()) / max(float(context_y.std()), 1e-8)).reshape(-1, 1)
    if transform is not None:
        target = transform.transform(target)
        expected_context = transform.transform(expected_context)
    target = torch.from_numpy(np.asarray(target, np.float32).reshape(1, -1))
    assert torch.isfinite(target).all()
    np.testing.assert_allclose(batch.y_context[0].cpu().numpy().ravel(), expected_context.ravel(), rtol=1e-5, atol=1e-6)
    return batch, target, query


def batch_logits(estimator, batch):
    estimator.raw_space_bardist_ = batch.raw_space_bardist
    estimator.znorm_space_bardist_ = batch.znorm_space_bardist
    estimator.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs, performance_options=PERFORMANCE)
    estimator.executor_.use_torch_inference_mode(use_inference=False)
    # Avoid inverse-transformed bar borders here: loss targets are in the
    # model's own transformed coordinate system, like its observed targets.
    device_type = next(estimator.models_[0].parameters()).device.type
    outputs = list(estimator.executor_.iter_outputs(batch.X_query, autocast=torch.is_autocast_enabled(device_type), task_type="regression"))
    assert len(outputs) == 1
    logits, _ = outputs[0]
    assert logits.ndim == 3 and logits.shape[1] == 1
    return (logits.float() / estimator.softmax_temperature).permute(1, 0, 2)


def loss(estimator, batch, target, device):
    return _compute_regression_loss(logits_BQL=batch_logits(estimator, batch), targets_BQ=target.to(device),
        bardist_loss_fn=batch.znorm_space_bardist, ce_loss_weight=0., crps_loss_weight=1., mse_loss_weight=1.)


def loss_parts(estimator, batch, target, device):
    """Preserve the original objective/gradient; measure components without a graph."""
    logits = batch_logits(estimator, batch)
    targets = target.to(device)
    options = dict(targets_BQ=targets, bardist_loss_fn=batch.znorm_space_bardist, ce_loss_weight=0.)
    total = _compute_regression_loss(logits_BQL=logits, crps_loss_weight=1., mse_loss_weight=1., **options)
    with torch.no_grad():
        crps = _compute_regression_loss(logits_BQL=logits.detach(), crps_loss_weight=1., mse_loss_weight=0., **options)
        mse = _compute_regression_loss(logits_BQL=logits.detach(), crps_loss_weight=0., mse_loss_weight=1., **options)
        torch.testing.assert_close(total.detach(), crps + mse, rtol=1e-6, atol=1e-7)
    return total, float(crps), float(mse)


def save_torch(value, path, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint(model, original, path):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    state.update({k: v for k, v in original["state_dict"].items() if k.startswith("criterion.")})
    assert state.keys() == original["state_dict"].keys()
    # Preserve the complete original inference configuration, including all
    # four X/y transform combinations and GPU preprocessing.
    value = dict(state_dict=state, config=original["config"], architecture_name=original["architecture_name"],
                 inference_config=copy.deepcopy(original["inference_config"]))
    save_torch(value, path)


def order_for_epoch(count, epoch, seed=0):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, 164000])).permutation(count)


def seed_for_step(step, seed=0):
    return int(np.random.SeedSequence([seed, step, 2048]).generate_state(1)[0])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
