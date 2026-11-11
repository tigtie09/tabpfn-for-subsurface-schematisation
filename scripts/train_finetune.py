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
"""Fine-tune on train16000; select on vali4000; never select using test/field outcomes."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(TABPFN_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", MPLBACKEND="Agg")
import numpy as np
import torch
from tabpfn.finetuning._torch_compat import sdpa_kernel_context
import finetune_core as core
from tabpfn_cpt_model import predict_source
from tabpfn_cpt_protocol import sha256, write_json
from common import atomic_json, error_metrics, load_data

ROOT = Path(__file__).resolve().parents[1]


def validation(epoch, infer, model, source, truth, output, checkpoint_path):
    directory = output / "validation" / f"epoch_{epoch}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(checkpoint_path)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        assert summary["checkpoint_sha256"] == checkpoint_hash and summary["n_cases"] == 4000
        for name, digest in summary["files"].items():
            assert sha256(directory / name) == digest
        return summary
    model.eval()
    before = core.parameter_digest(model)
    records, hashes = [], {}
    for first in range(0, len(source), 4):
        path = directory / f"block_{first:04d}.npz"
        indices = list(range(first, min(first + 4, len(source))))
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                assert str(saved["checkpoint_sha256"]) == checkpoint_hash
                np.testing.assert_array_equal(saved["indices"], indices)
                predictions = saved["predictions"]
        else:
            predictions = []
            for i in indices:
                src = source[i, :, :, 0]
                prediction, diagnostic, _ = predict_source(infer, src, src != 0, torch)
                if diagnostic["warnings"]:
                    print(json.dumps(dict(validation_epoch=epoch, case=i, warnings=diagnostic["warnings"])), flush=True)
                predictions.append(prediction)
            predictions = np.stack(predictions)
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, indices=indices, predictions=predictions, checkpoint_sha256=np.asarray(checkpoint_hash))
            temporary.replace(path)
        assert predictions.shape == (len(indices), 32, 512) and np.isfinite(predictions).all()
        for i, prediction in zip(indices, predictions):
            src, tar = source[i, :, :, 0], truth[i, :, :, 0]
            scores = {
                f"{scope}_{key}": value
                for scope, mask in (
                    ("all_pixels", np.ones_like(src, dtype=bool)),
                    ("unobserved_pixels", src == 0),
                )
                for key, value in error_metrics(prediction, tar, mask).items()
            }
            records.append(dict(case_index=i, **scores))
        hashes[path.name] = sha256(path)
        if (first + len(indices)) % 100 == 0:
            progress = dict(status="validation", epoch=epoch, completed_cases=len(records), n_cases=len(source))
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert core.parameter_digest(model) == before
    with (directory / "metrics_by_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    hashes["metrics_by_case.csv"] = sha256(directory / "metrics_by_case.csv")
    metrics = {key: float(np.mean([r[key] for r in records])) for key in records[0] if key not in ("case_index", "all_pixels_n_pixels", "unobserved_pixels_n_pixels")}
    summary = dict(status="completed", epoch=epoch, n_cases=len(records), checkpoint=str(checkpoint_path),
        checkpoint_sha256=checkpoint_hash, model_parameters_sha256=before, metrics=metrics, files=hashes)
    atomic_json(summary_path, summary)
    print(json.dumps(dict(validation_epoch=epoch, **metrics)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if sys.flags.optimize:
        raise ValueError("Assertions must be enabled")
    config = json.loads(args.config.read_text())
    settings = config["training"]
    assert settings["n_train"] == 16000 and settings["n_validation"] == 4000
    assert config["seed"] in (0, 1) and config["inference"]["n_estimators"] == 8
    assert settings["max_epochs"] == 5 and settings["query_pixels"] == 2048
    assert config["validation"]["no_test_or_real_scores_for_selection"]
    versions = {name: importlib.metadata.version(name) for name in config["environment"]}
    assert versions == config["environment"]
    data_files = {}
    for split, expected_count in (("train", 16000), ("vali", 4000)):
        source, truth, _, manifest = load_data(config["prepared"][split])
        assert len(source) == expected_count
        assert manifest["config"]["split"] == split
        assert manifest["output_sha256"] == manifest["config"]["reference_sha256"]
        data_files[split] = manifest["output_sha256"]
    assert sha256(ROOT / config["initial_model"]) == config["initial_model_sha256"]
    sources = [Path(__file__), Path(__file__).with_suffix(".py.lock"),
        Path(__file__).with_name("finetune_core.py"), Path(__file__).with_name("tabpfn_cpt_model.py"),
        Path(__file__).with_name("tabpfn_cpt_protocol.py"), Path(__file__).with_name("common.py"),
        Path(__file__).with_name("check_finetune.py"), Path(__file__).with_name("check_finetune.py.lock")]
    contract = dict(config=config, config_sha256=sha256(args.config), packages=versions,
        input_sha256=data_files, code_sha256={p.name: sha256(p) for p in sources})
    output = args.output_dir.resolve()
    if args.resume:
        recorded = json.loads((output / "run.json").read_text())
        assert all(recorded[k] == value for k, value in contract.items()), "Resume contract changed"
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(), **contract))
    with (output / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "selection.json").exists():
            selected = json.loads((output / "selection.json").read_text())
            assert sha256(selected["checkpoint"]) == selected["checkpoint_sha256"]
            print("Training already completed; selected checkpoint hash verified", flush=True)
            return
        try:
            smoke = output / "gpu_check"
            if not (smoke / "verification.json").exists():
                subprocess.run(["uv", "run", "--locked", str(Path(__file__).with_name("check_finetune.py")),
                    "--device", "cuda", "--config", str(args.config), "--output-dir", str(smoke)], check=True)
            checked = json.loads((smoke / "verification.json").read_text())
            assert checked["status"] == "passed" and checked['config_sha256'] == sha256(args.config)
            for name, expected in checked["source_sha256"].items():
                assert sha256(Path(__file__).with_name(name)) == expected
            core.initialize("cuda", config['seed'])
            started = time.perf_counter()
            hardware = dict(gpu=torch.cuda.get_device_name(), torch_cuda=torch.version.cuda, threads=torch.get_num_threads())
            arrays = {}
            for split, relative in config["prepared"].items():
                arrays[split] = tuple(np.load(ROOT / relative / name, mmap_mode="r", allow_pickle=False)
                                      for name in ("src_images.npy", "tar_images.npy"))
            train, infer, model, families = core.estimators(ROOT / config["initial_model"], "cuda")
            original = torch.load(ROOT / config["initial_model"], map_location="cpu", weights_only=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"])
            total = settings["n_train"] * settings["max_epochs"]
            step, updates = 0, 0
            loss_sums = {}
            checkpoint_dir = output / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            state_path = output / "state_last.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                scaler.load_state_dict(state["scaler"])
                core.restore_rng(state["rng"])
                step, updates = state["step"], state["optimizer_updates"]
                loss_sums = state['loss_sums']
                del state
            def save_state():
                core.save_torch(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                     rng=core.rng_state(), step=step, optimizer_updates=updates, loss_sums=loss_sums), state_path, replace=True)
            initial = checkpoint_dir / "epoch_0.ckpt"
            if not initial.exists():
                assert step == 0
                core.checkpoint(model, original, initial)
            history = [validation(0, infer, model, *arrays["vali"], output, initial)]
            log = output / "training_log.jsonl"
            for epoch in range(1, settings["max_epochs"] + 1):
                order = core.order_for_epoch(settings["n_train"], epoch, config["seed"])
                model.train()
                for position in range(max(0, step - (epoch - 1) * settings["n_train"]), settings["n_train"]):
                    i = int(order[position])
                    src, tar = (a[i, :, :, 0] for a in arrays["train"])
                    batch, targets, _ = core.make_batch(train, src, tar, families[step % 4], core.seed_for_step(step, config['seed']), settings["query_pixels"])
                    warmup = settings["warmup_steps"]
                    fraction = (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
                    lr = settings["learning_rate"] * fraction
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", enabled=settings["amp"]), sdpa_kernel_context():
                        value, crps, mse = core.loss_parts(train, batch, targets, "cuda")
                    assert torch.isfinite(value), (step, i, value)
                    with sdpa_kernel_context():
                        scaler.scale(value).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
                    scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    successful = scaler.get_scale() >= scale
                    updates += int(successful)
                    sums = loss_sums.setdefault(str(epoch), [0, 0., 0.])
                    sums[0] += 1; sums[1] += crps; sums[2] += mse
                    step += 1
                    if step % 100 == 0 or not successful:
                        record = dict(status="training", epoch=epoch, step=step, total_steps=total, optimizer_updates=updates,
                            loss=float(value.detach()), gradient_norm=float(norm) if torch.isfinite(norm) else None,
                            learning_rate=lr, last_update_successful=successful, peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                            crps=crps, mse=mse, epoch_mean_crps=sums[1]/sums[0], epoch_mean_mse=sums[2]/sums[0],
                            invocation_seconds=time.perf_counter()-started, updated_at_utc=datetime.now(timezone.utc).isoformat())
                        with log.open("a") as stream:
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                        atomic_json(output / "progress.json", record)
                        print(json.dumps(record), flush=True)
                    if step % settings["checkpoint_every"] == 0:
                        save_state()
                path = checkpoint_dir / f"epoch_{epoch}.ckpt"
                if not path.exists():
                    assert step == epoch * settings["n_train"]
                    core.checkpoint(model, original, path)
                save_state()
                history.append(validation(epoch, infer, model, *arrays["vali"], output, path))
                atomic_json(output/'loss_components.json', dict(training_seed=config['seed'], scale='context-fitted internal target transform; averages over all attempted updates', epochs={e:dict(n_steps=v[0], crps=v[1]/v[0], mse=v[2]/v[0]) for e,v in loss_sums.items()}))
            best = min(history[1:], key=lambda r: (r["metrics"]["unobserved_pixels_mae"], r["epoch"]))
            best_with_zero = min(history, key=lambda r: (r["metrics"]["unobserved_pixels_mae"], r["epoch"]))
            selection = dict(status="completed", selection_uses="vali4000 only", **{k: best[k] for k in ("epoch", "checkpoint", "checkpoint_sha256")},
                validation_mae=best["metrics"]["unobserved_pixels_mae"], initial_validation_mae=history[0]["metrics"]["unobserved_pixels_mae"],
                best_epoch_if_no_update_is_allowed=best_with_zero["epoch"], attempted_updates=step, successful_updates=updates,
                validation_summaries={str(r["epoch"]): sha256(output / "validation" / f'epoch_{r["epoch"]}' / "summary.json") for r in history},
                validation_history=[dict(epoch=r["epoch"], metrics=r["metrics"], checkpoint_sha256=r["checkpoint_sha256"]) for r in history],
                training_seed=config['seed'], hardware=hardware, invocation_seconds=time.perf_counter()-started, peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/2**20)
            write_json(output / "selection.json", selection)
            atomic_json(output / "progress.json", dict(status="completed", epoch=best["epoch"], step=step, selected_validation_mae=selection["validation_mae"]))
        except BaseException as error:
            atomic_json(output / "progress.json", dict(status="failed", error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    main()
