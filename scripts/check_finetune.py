# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "tabpfn==8.5.0", "torch==2.11.0+cu128", "numpy==2.4.6",
#   "scipy==1.17.1", "scikit-learn==1.9.0", "pandas==3.0.5", "matplotlib==3.10.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Check four training transform families, label isolation, and exact resumption."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(TABPFN_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
import numpy as np
import torch
from tabpfn.finetuning._torch_compat import sdpa_kernel_context
import finetune_core as core
from tabpfn_cpt_protocol import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise ValueError("Assertions must be enabled")
    config = json.loads(args.config.read_text())
    core.initialize(args.device, config["seed"])
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / config["initial_model"]
    assert sha256(checkpoint) == config["initial_model_sha256"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source = np.load(root / config["prepared"]["train"] / "src_images.npy", mmap_mode="r")
    truth = np.load(root / config["prepared"]["train"] / "tar_images.npy", mmap_mode="r")
    train, infer, model, families = core.estimators(checkpoint, args.device)
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    initial = core.parameter_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
    amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    query_count = 2048 if amp else 32
    def update(batch, targets):
        # Dynamic AMP scaling may initially skip overflowing gradients. Require
        # an actual finite update for each tested transform family.
        for attempt in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(args.device, enabled=amp), sdpa_kernel_context():
                value, crps, mse = core.loss_parts(train, batch, targets, args.device)
            assert torch.isfinite(value)
            with sdpa_kernel_context():
                scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                assert torch.isfinite(norm) and float(norm) > 0
                return value, norm, attempt
        raise RuntimeError("AMP failed to produce finite gradients in 12 attempts")
    reports = []
    saved_inference = repr(infer.inference_config_)
    for j, family in enumerate(families):
        src, tar = source[j, :, :, 0], truth[j, :, :, 0]
        batch, targets, query = core.make_batch(train, src, tar, family, core.seed_for_step(j, config["seed"]), query_count)
        changed_truth = tar.copy()
        changed_truth[src == 0] += .37
        changed, changed_targets, same_query = core.make_batch(train, src, changed_truth, family, core.seed_for_step(j, config["seed"]), query_count)
        np.testing.assert_array_equal(query, same_query)
        for field in ("X_context", "X_query", "y_context"):
            for a, b in zip(getattr(batch, field), getattr(changed, field)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not torch.equal(targets, changed_targets)
        model.train()
        before = core.rng_state()
        with torch.autocast(args.device, enabled=amp), sdpa_kernel_context():
            expected = core.loss(train, batch, targets, args.device).detach()
        after = core.rng_state()
        core.restore_rng(before)
        with torch.autocast(args.device, enabled=amp), sdpa_kernel_context():
            actual, crps, mse = core.loss_parts(train, batch, targets, args.device)
        torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)
        assert torch.equal(core.rng_state()['torch'], after['torch'])
        del actual, expected
        core.restore_rng(before)
        value, norm, skipped = update(batch, targets)
        reports.append(dict(family=j, x=family[0].name, y=family[1], loss=float(value.detach()), gradient_norm=float(norm), amp_skipped_attempts=skipped, objective_matches_reference=True, crps_before_update=crps, mse_before_update=mse))
        print(reports[-1], flush=True)
        if j == 1:
            core.save_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=core.rng_state()), output / "resume.pt")
    uninterrupted = core.parameter_digest(model)
    assert uninterrupted != initial
    state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    core.restore_rng(state["rng"])
    for j in (2, 3):
        batch, targets, _ = core.make_batch(train, source[j, :, :, 0], truth[j, :, :, 0], families[j], core.seed_for_step(j, config["seed"]), query_count)
        update(batch, targets)
    assert core.parameter_digest(model) == uninterrupted, "Resumed training differs"
    assert repr(infer.inference_config_) == saved_inference
    core.checkpoint(model, original, output / "trained.ckpt")
    _, loaded_infer, loaded_model, _ = core.estimators(output / "trained.ckpt", args.device)
    assert core.parameter_digest(loaded_model) == uninterrupted
    X, y = core.training_arrays(source[0, :, :, 0], source[0, :, :, 0] != 0)
    predictions = []
    for estimator, weights in ((infer, model), (loaded_infer, loaded_model)):
        weights.eval()
        estimator.fit(X, y)
        predictions.append(estimator.predict(core.GRID[::128], output_type="median"))
    np.testing.assert_array_equal(*predictions)
    record = dict(status="passed", device=args.device, training_seed=config["seed"], config_sha256=sha256(args.config), training_amp=amp, query_pixels=query_count,
        peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if amp else None, initial_parameters_sha256=initial,
        trained_parameters_sha256=uninterrupted, exact_resume=True, exact_saved_prediction=True,
        heldout_labels_cannot_change_context_or_query_features=True, four_transform_families=reports,
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in ("finetune_core.py", "check_finetune.py")})
    write_json(output / "verification.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
