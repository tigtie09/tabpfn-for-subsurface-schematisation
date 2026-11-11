"""Standalone synthetic prediction runs using the manuscript's numerical helpers."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from common import (
    ROOT,
    LEVELS,
    QUANTILES,
    COUNTS,
    load_data,
    error_metrics,
    observation_mask,
    sha256,
    write_csv,
    write_json,
)
from tabpfn_cpt_protocol import training_arrays, query_grid
from interval_metrics import normal_bounds, score_intervals


def kriging_intervals(source, params, prediction, index):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    X, y = training_arrays(source, source != 0)
    kernel = RBF([params["length_scale_x"], params["length_scale_z"]]) + WhiteKernel(
        params["noise_level"]
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=params["alpha"],
        normalize_y=True,
        optimizer=None,
        random_state=index,
    )
    model.fit(X.astype(np.float64), y.astype(np.float64))
    mean, std = model.predict(query_grid().astype(np.float64), return_std=True)
    mean, std = mean.reshape(32, 512), std.reshape(32, 512)
    np.testing.assert_allclose(mean, prediction, rtol=0, atol=1e-6)
    return normal_bounds(mean, std, LEVELS)


def main(method):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="SchemaGAN inference seed; TabPFN uses seed 0",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/tabpfn-v3-regressor-v3_default.ckpt",
    )
    parser.add_argument("--layouts", type=Path)
    parser.add_argument("--count", type=int, choices=COUNTS)
    parser.add_argument("--intervals", action="store_true")
    args = parser.parse_args()
    if bool(args.layouts) != bool(args.count):
        parser.error("--layouts and --count must be supplied together")
    if method != "schemagan" and args.seed != 0:
        parser.error(
            "Only SchemaGAN uses --seed; the other methods retain the manuscript settings"
        )
    if args.intervals and (method not in ("tabpfn", "kriging") or args.count):
        parser.error(
            "Intervals here are for baseline TabPFN/Kriging; aggregate SchemaGAN draws separately"
        )
    source, truth, sections, manifest = load_data(args.data, args.limit)
    layout = None
    if args.layouts:
        with np.load(args.layouts / "layouts.npz", allow_pickle=False) as saved:
            layout = (saved["columns"], saved["depths"])
        info = json.loads((args.layouts / "manifest.json").read_text())
        if (
            info["source_sha256"] != manifest["output_sha256"]["src_images.npy"]
            or info["layouts_sha256"] != sha256(args.layouts / "layouts.npz")
            or len(layout[0]) < len(source)
        ):
            raise ValueError("Layouts do not match the prepared baseline")
    model_hash = None
    if method == "tabpfn":
        model_hash = sha256(args.checkpoint)
        settings = json.loads((ROOT / "configs/tabpfn.json").read_text())
        if (
            args.checkpoint.name == Path(settings["model"]["path"]).name
            and model_hash != settings["model"]["sha256"]
        ):
            raise ValueError("Default TabPFN checkpoint checksum differs")
    elif method == "schemagan":
        model_hash = sha256(ROOT / "models/schemaGAN.h5")
        if (
            model_hash
            != "b4834d8749a738538fd02b494dd4951197f8ee7f47f95f2be0add53bc6e5a987"
        ):
            raise ValueError("SchemaGAN checkpoint checksum differs")
    contract = dict(
        method=method,
        n_sections=len(source),
        inference_seed=args.seed,
        checkpoint_sha256=model_hash,
        data_sha256=manifest["output_sha256"],
        count=args.count,
        intervals=args.intervals,
        layouts_sha256=sha256(args.layouts / "layouts.npz") if args.layouts else None,
        script_sha256={
            p.name: sha256(p) for p in sorted((ROOT / "scripts").glob("*.py"))
        },
    )
    if args.output.exists():
        completed = args.output / "run.json"
        if completed.exists():
            old = json.loads(completed.read_text())
            if old["contract"] == contract and all(
                sha256(args.output / n) == h for n, h in old["files"].items()
            ):
                print("Completed run verified:", args.output)
                return
        raise FileExistsError(f"Existing run is incomplete or differs: {args.output}")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    if method == "tabpfn":
        from tabpfn_cpt_model import initialize_torch, make_regressor, parameter_digest

        torch = initialize_torch()
        estimator, weights = make_regressor(args.checkpoint, torch)
        before = parameter_digest(weights)
    elif method == "schemagan":
        from schemagan_model import load_model

        estimator, _, _ = load_model(args.seed)
    elif method == "kriging":
        from kriging_model import fit_predict
    else:
        from nearest_model import nearest_grid
    predictions = np.lib.format.open_memmap(
        args.output / "predictions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(source), 32, 512),
    )
    means = (
        np.lib.format.open_memmap(
            args.output / "means.npy",
            mode="w+",
            dtype=np.float32,
            shape=predictions.shape,
        )
        if method == "tabpfn"
        else None
    )
    intervals = (
        np.lib.format.open_memmap(
            args.output / "bounds.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(source), 4, 2, 32, 512),
        )
        if args.intervals
        else None
    )
    rows, interval_rows, parameters = [], [], []
    for i, (baseline_source, target) in enumerate(zip(source, truth)):
        x = baseline_source
        if layout is not None:
            observed = observation_mask(layout[0][i], layout[1][i], args.count)
            x = np.where(observed, target, 0).astype(np.float32)
        observed = x != 0
        np.testing.assert_array_equal(x[observed], target[observed])
        bounds = None
        if method == "tabpfn":
            X, y = training_arrays(x, observed)
            estimator.fit(X, y)
            output = estimator.predict(
                query_grid(), output_type="main", quantiles=list(QUANTILES)
            )
            prediction = np.asarray(output["median"], np.float32).reshape(32, 512)
            mean = np.asarray(output["mean"], np.float32).reshape(32, 512)
            means[i] = mean
            if args.intervals:
                quantiles = np.stack(output["quantiles"]).reshape(8, 32, 512)
                bounds = np.stack([quantiles[[3 - j, 4 + j]] for j in range(4)])
        elif method == "schemagan":
            normalized = 2 * (x[None, :, :, None] / 4.3) - 1
            generated = estimator.predict_on_batch(normalized)
            prediction = ((generated + 1) * (4.3 / 2))[0, :, :, 0]
        elif method == "kriging":
            prediction, _, params = fit_predict(x, observed, i)
            parameters.append(dict(array_index=i, **params))
            if args.intervals:
                bounds = kriging_intervals(x, params, prediction, i)
        else:
            prediction, _ = nearest_grid(x, observed)
        predictions[i] = prediction
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
        points = [
            (
                "median"
                if method == "tabpfn"
                else "single_draw"
                if method == "schemagan"
                else "mean"
                if method == "kriging"
                else "nearest",
                prediction,
            )
        ]
        if means is not None:
            points.append(("mean", mean))
        for point, values in points:
            for scope, mask in masks:
                rows.append(
                    dict(
                        array_index=i,
                        section_id=sections[i]["section_id"],
                        method=method,
                        point=point,
                        scope=scope,
                        **error_metrics(values, target, mask),
                    )
                )
        if bounds is not None:
            intervals[i] = bounds
            interval_rows.extend(
                dict(
                    array_index=i,
                    section_id=sections[i]["section_id"],
                    method=method,
                    **r,
                )
                for r in score_intervals(target, observed, bounds, LEVELS)
            )
        if (i + 1) % 100 == 0 or i + 1 == len(source):
            print(f"{method} seed={args.seed}: {i + 1}/{len(source)}", flush=True)
    predictions.flush()
    if means is not None:
        means.flush()
        if parameter_digest(weights) != before:
            raise ValueError("Inference changed TabPFN weights")
    if intervals is not None:
        intervals.flush()
        write_csv(args.output / "interval_metrics.csv", interval_rows)
    write_csv(args.output / "metrics.csv", rows)
    if parameters:
        write_csv(args.output / "parameters.csv", parameters)
    files = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "tabpfn", "torch", "tensorflow-cpu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(
        args.output / "run.json",
        dict(
            status="completed",
            contract=contract,
            files=files,
            environment=packages,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
