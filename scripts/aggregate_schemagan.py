"""Score individual draws before averaging; also compute predictive means and intervals."""

import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import numpy as np
from common import (
    ROOT,
    LEVELS,
    load_data,
    error_metrics,
    observation_mask,
    sha256,
    write_csv,
    write_json,
)
from interval_metrics import normal_bounds, score_intervals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/test")
    p.add_argument("--draws", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(100)))
    p.add_argument("--limit", type=int)
    p.add_argument("--layouts", type=Path)
    p.add_argument("--count", type=int, choices=(3, 6, 12, 24, 48))
    p.add_argument("--intervals", action="store_true")
    a = p.parse_args()
    if len(a.seeds) < 2 or len(a.seeds) != len(set(a.seeds)):
        p.error("Need at least two distinct seeds")
    if bool(a.layouts) != bool(a.count) or (a.intervals and a.count):
        p.error(
            "Supply count and layouts together; intervals are evaluated at baseline only"
        )
    source, truth, sections, manifest = load_data(a.data, a.limit)
    layout = None
    if a.layouts:
        with np.load(a.layouts / "layouts.npz", allow_pickle=False) as f:
            layout = (f["columns"], f["depths"])
    draws = []
    inputs = {}
    for seed in a.seeds:
        directory = a.draws / f"seed_{seed}"
        run = json.loads((directory / "run.json").read_text())
        c = run["contract"]
        if (
            c["data_sha256"] != manifest["output_sha256"]
            or c["count"] != a.count
            or c["inference_seed"] != seed
        ):
            raise ValueError("Draw settings do not match requested data/count/seed")
        path = directory / "predictions.npy"
        digest = sha256(path)
        if digest != run["files"]["predictions.npy"]:
            raise ValueError(f"Changed draw: {path}")
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(x) != len(source):
            raise ValueError("Draw count does not match prepared data/limit")
        draws.append(x)
        inputs[str(seed)] = digest
    contract = dict(
        seeds=a.seeds,
        n_sections=len(source),
        data_sha256=manifest["output_sha256"],
        input_predictions_sha256=inputs,
        count=a.count,
        intervals=a.intervals,
        layouts_sha256=sha256(a.layouts / "layouts.npz") if a.layouts else None,
        script_sha256={
            p.name: sha256(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/common.py",
                ROOT / "scripts/interval_metrics.py",
            )
        },
    )
    if (a.output / "run.json").exists():
        previous = json.loads((a.output / "run.json").read_text())
        if previous.get("contract") == contract and all(
            sha256(a.output / name) == h for name, h in previous["files"].items()
        ):
            print("Completed aggregation verified:", a.output)
            return
        raise ValueError("Existing aggregation settings or files differ")
    a.output.mkdir(parents=True, exist_ok=False)
    shape = (len(source), 32, 512)
    means = np.lib.format.open_memmap(
        a.output / "means.npy", mode="w+", dtype=np.float64, shape=shape
    )
    medians = np.lib.format.open_memmap(
        a.output / "medians.npy", mode="w+", dtype=np.float64, shape=shape
    )
    bounds_out = (
        np.lib.format.open_memmap(
            a.output / "bounds.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(source), 4, 2, 32, 512),
        )
        if a.intervals
        else None
    )
    records = []
    interval_records = []
    for i, t in enumerate(truth):
        observed = (
            source[i] != 0
            if layout is None
            else observation_mask(layout[0][i], layout[1][i], a.count)
        )
        samples = np.stack([d[i] for d in draws]).astype(np.float64)
        mean, median = samples.mean(axis=0), np.median(samples, axis=0)
        means[i] = mean
        medians[i] = median
        masks = [
            ("all_pixels", np.ones_like(observed)),
            ("unobserved_pixels", ~observed),
        ]
        if layout is not None:
            masks.append(
                (
                    "common_unobserved_pixels",
                    ~observation_mask(layout[0][i], layout[1][i], 48),
                )
            )
        for scope, mask in masks:
            errors = [error_metrics(s, t, mask) for s in samples]
            average = {
                k: float(np.mean([e[k] for e in errors]))
                for k in ("mae", "mse", "rmse")
            }
            for point, values in [
                ("mean_draw_error", average),
                ("mean", error_metrics(mean, t, mask)),
                ("median", error_metrics(median, t, mask)),
            ]:
                records.append(
                    dict(
                        array_index=i,
                        section_id=sections[i]["section_id"],
                        method="schemagan",
                        point=point,
                        scope=scope,
                        n_pixels=int(mask.sum()),
                        **{k: values[k] for k in ("mae", "mse", "rmse")},
                    )
                )
        if a.intervals:
            bounds = normal_bounds(mean, samples.std(axis=0, ddof=1), LEVELS)
            bounds_out[i] = bounds
            interval_records.extend(
                dict(
                    array_index=i,
                    section_id=sections[i]["section_id"],
                    method="schemagan",
                    **r,
                )
                for r in score_intervals(t, observed, bounds, LEVELS)
            )
        if (i + 1) % 100 == 0 or i + 1 == len(source):
            print(f"Aggregated {i + 1}/{len(source)}", flush=True)
    means.flush()
    medians.flush()
    write_csv(a.output / "metrics.csv", records)
    if bounds_out is not None:
        bounds_out.flush()
        write_csv(a.output / "interval_metrics.csv", interval_records)
    write_json(
        a.output / "run.json",
        dict(
            status="completed",
            contract=contract,
            n_draws=len(draws),
            seeds=a.seeds,
            data_sha256=manifest["output_sha256"],
            input_predictions_sha256=inputs,
            files={p.name: sha256(p) for p in a.output.iterdir() if p.is_file()},
            aggregation="Score each draw before averaging errors; predictive mean/median scored separately; intervals use sample SD, not standard error",
        ),
    )


if __name__ == "__main__":
    main()
