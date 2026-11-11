"""Summarise newly generated baseline/count predictions and the distance experiment."""

import os

for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[k] = "1"
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from common import ROOT, load_data, sha256

METHODS = ("tabpfn", "schemagan", "kriging", "nearest_neighbour")


def table1(frame):
    frame = frame[frame.scope == "all_pixels"]
    result = []
    for metric in ("mae", "rmse"):
        baseline = (
            frame[frame.method == "schemagan"]
            .sort_values("array_index")[metric]
            .to_numpy()
        )
        for method in METHODS:
            selected = frame[frame.method == method].sort_values("array_index")
            v = selected[metric].to_numpy()
            n = len(v)
            np.testing.assert_array_equal(selected.array_index, np.arange(n))
            d = v - baseline
            radius = 1.96 * d.std(ddof=1) / np.sqrt(n)
            result.append(
                dict(
                    metric=metric,
                    method=method,
                    n=n,
                    mean=float(v.mean()),
                    sd=float(v.std(ddof=1)),
                    difference=float(d.mean()),
                    ci_low=float(d.mean() - radius),
                    ci_high=float(d.mean() + radius),
                    win_rate=float(np.mean(d < 0)),
                )
            )
    return pd.DataFrame(result)


def verified(directory, name):
    report = json.loads((directory / "run.json").read_text())
    path = directory / name
    if report["status"] != "completed" or sha256(path) != report["files"][name]:
        raise ValueError(f"Changed/incomplete result: {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Directory containing tabpfn, schemagan_summary, kriging, nearest_neighbour",
    )
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--counts", type=Path, help="Directory containing n_3 ... n_48")
    p.add_argument(
        "--distance",
        action="store_true",
        help="Read all 100 saved draw arrays; can take substantial I/O",
    )
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source, truth, _, manifest = load_data(a.data, a.limit)
    frames = []
    intervals = []
    for method, folder, point in [
        ("tabpfn", "tabpfn", "median"),
        ("schemagan", "schemagan_summary", "mean_draw_error"),
        ("kriging", "kriging", "mean"),
        ("nearest_neighbour", "nearest_neighbour", "nearest"),
    ]:
        directory = a.baseline / folder
        run = json.loads((directory / "run.json").read_text())
        if (
            run.get("data_sha256", run.get("contract", {}).get("data_sha256"))
            != manifest["output_sha256"]
        ):
            raise ValueError(f"Run uses different prepared data: {directory}")
        f = pd.read_csv(
            verified(directory, "metrics.csv"), float_precision="round_trip"
        )
        frames.append(f[f.point == point].drop(columns=["point"]))
        if (directory / "interval_metrics.csv").exists():
            intervals.append(
                pd.read_csv(
                    verified(directory, "interval_metrics.csv"),
                    float_precision="round_trip",
                )
            )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(a.output / "baseline_by_section.csv", index=False)
    table1(frame).to_csv(a.output / "table1.csv", index=False)
    if intervals:
        f = pd.concat(intervals, ignore_index=True)
        f = f[f.scope == "unobserved_pixels"]
        f.to_csv(a.output / "intervals_by_section.csv", index=False)
        f.groupby(["level", "method"])[
            ["coverage", "width_mean", "interval_score"]
        ].mean().to_csv(a.output / "table2.csv")
    if a.counts:
        counts = []
        for count in (3, 6, 12, 24, 48):
            for method, folder, point in [
                ("tabpfn", "tabpfn", "median"),
                ("schemagan", "schemagan_summary", "mean_draw_error"),
            ]:
                f = pd.read_csv(
                    verified(a.counts / f"n_{count}" / folder, "metrics.csv"),
                    float_precision="round_trip",
                )
                f = f[
                    (f.point == point) & (f.scope == "common_unobserved_pixels")
                ].copy()
                f["count"] = count
                counts.append(f[["array_index", "method", "count", "mae"]])
        f = pd.concat(counts, ignore_index=True)
        f.to_csv(a.output / "counts_by_section.csv", index=False)
        g = (
            f.groupby(["method", "count"])
            .mae.agg(["mean", "std", "count"])
            .rename(columns={"count": "n"})
        )
        g["ci_low"] = g["mean"] - 1.96 * g["std"] / np.sqrt(g["n"])
        g["ci_high"] = g["mean"] + 1.96 * g["std"] / np.sqrt(g["n"])
        g.to_csv(a.output / "counts_summary.csv")
    if a.distance:
        tab = np.load(verified(a.baseline / "tabpfn", "predictions.npy"), mmap_mode="r")
        ensemble = json.loads((a.baseline / "schemagan_summary/run.json").read_text())
        draws = [
            np.load(
                verified(a.baseline / "schemagan" / f"seed_{s}", "predictions.npy"),
                mmap_mode="r",
            )
            for s in ensemble["seeds"]
        ]
        values = np.full((2, len(source), 130), np.nan, np.float64)
        for i, (x, t) in enumerate(zip(source, truth)):
            distance = distance_transform_edt(x == 0)
            bins = np.ceil(distance / 4).astype(int)
            target = t.astype(np.float64)
            # Mean absolute error can be averaged pixelwise over draws before
            # bin reduction; RMSE cannot, and is not computed by this shortcut.
            errors = [
                np.abs(tab[i].astype(np.float64) - target),
                np.mean(
                    [np.abs(d[i].astype(np.float64) - target) for d in draws], axis=0
                ),
            ]
            for b in np.unique(bins):
                mask = bins == b
                for m, error in enumerate(errors):
                    values[m, i, b] = error[mask].mean()
        records = []
        for m, series in [
            ("tabpfn", values[0]),
            ("schemagan", values[1]),
            ("tabpfn_minus_schemagan", values[0] - values[1]),
        ]:
            for b in range(130):
                v = series[:, b]
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                mean = float(v.mean())
                sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                radius = 1.96 * sd / np.sqrt(len(v))
                records.append(
                    dict(
                        bin=b,
                        distance_low=max(0, 4 * (b - 1)),
                        distance_high=4 * b,
                        method=m,
                        metric="mae",
                        n_sections=len(v),
                        mean=mean,
                        section_sd=sd,
                        ci_low=mean - radius,
                        ci_high=mean + radius,
                    )
                )
        pd.DataFrame(records).to_csv(a.output / "distance_summary.csv", index=False)
    print(a.output)


if __name__ == "__main__":
    main()
