"""Reproduce Table B.1 and its SchemaGAN output-summary check from baseline runs.

All in-grid pairs at each separation are used, including input pixels. Ratios
are formed within sections; SchemaGAN draws are averaged before the median
across sections. No inference, training, layer masks or distance scaling is used.
"""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from common import ROOT, load_data, sha256, write_csv, write_json

LAGS = [("V", lag) for lag in (1, 2, 4, 8)] + [
    ("H", lag) for lag in (1, 2, 4, 8, 16, 32, 64, 128)
]
METRICS = ("mean_absolute_difference", "median_absolute_difference")
METHODS = ("tabpfn", "schemagan", "kriging", "nearest_neighbour")
SOURCE_COMMIT = "22322a44e6b37dd2323be34450b15f2c30206416"


def descriptors(fields):
    """Mean and median absolute differences, with float64 subtraction."""
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim != 3 or fields.shape[1:] != (32, 512) or not len(fields):
        raise ValueError("Expected one or more 32 x 512 sections")
    if not np.isfinite(fields).all():
        raise ValueError("Non-finite section values")
    values = np.empty((len(fields), len(LAGS), len(METRICS)), np.float64)
    for index, (direction, lag) in enumerate(LAGS):
        if direction == "V":
            difference = np.abs(fields[:, lag:, :] - fields[:, :-lag, :])
        else:
            difference = np.abs(fields[:, :, lag:] - fields[:, :, :-lag])
        pairs = difference.reshape(len(fields), -1)
        values[:, index, 0] = pairs.mean(axis=1)
        values[:, index, 1] = np.median(pairs, axis=1)
    return values


def score_sections(fields, n, block_size):
    if not 1 <= n <= len(fields) or not 1 <= block_size <= 64:
        raise ValueError("Invalid section count or block size")
    values = np.empty((n, len(LAGS), len(METRICS)), np.float64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        values[start:stop] = descriptors(fields[start:stop])
    return values


def ratios_to_target(values, target):
    if values.shape != target.shape or not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError("Ratios require matching arrays and positive target descriptors")
    ratios = values / target
    if not np.isfinite(ratios).all() or np.any(ratios < 0):
        raise ValueError("Invalid reconstructed-to-target ratios")
    return ratios


def summaries(ratios):
    rows = []
    for method in METHODS:
        for index, (direction, lag) in enumerate(LAGS):
            for metric_index, metric in enumerate(METRICS):
                values = ratios[method][:, index, metric_index]
                q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
                rows.append(dict(
                    method=method, direction=direction, separation=lag,
                    descriptor=metric, n_sections=len(values), median=float(median),
                    q25=float(q25), q75=float(q75),
                    min=float(values.min()), max=float(values.max()),
                ))
    return rows


def compare_output_summaries(baseline, median_field, tabpfn):
    """Compare medians across sections, not the largest individual change."""
    old = np.median(baseline, axis=0)
    new = np.median(median_field, axis=0)
    tab = np.median(tabpfn, axis=0)
    rows, findings = [], {}
    for metric_index, metric in enumerate(METRICS):
        changes = np.abs(new[:, metric_index] - old[:, metric_index])
        largest = int(changes.argmax())
        findings[metric] = dict(
            max_absolute_change_in_reported_medians=float(changes.max()),
            largest_change_separation="".join(map(str, LAGS[largest])),
            max_absolute_sectionwise_change=float(np.max(
                np.abs(median_field[:, :, metric_index] - baseline[:, :, metric_index])
            )),
            tabpfn_below_one_all=bool(np.all(tab[:, metric_index] < 1)),
            schemagan_median_field_exceeds_tabpfn_all=bool(
                np.all(new[:, metric_index] > tab[:, metric_index])
            ),
            schemagan_median_field_above_one=[
                f"{direction}{lag}" for index, (direction, lag) in enumerate(LAGS)
                if new[index, metric_index] > 1
            ],
        )
        for index, (direction, lag) in enumerate(LAGS):
            rows.append(dict(
                descriptor=metric, direction=direction, separation=lag,
                baseline=float(old[index, metric_index]),
                pixelwise_median=float(new[index, metric_index]),
                change=float(new[index, metric_index] - old[index, metric_index]),
            ))
    return rows, findings


def read_run(directory):
    record = json.loads((directory / "run.json").read_text())
    if record.get("status") != "completed":
        raise ValueError(f"Incomplete run: {directory}")
    return record


def checked_array(directory, record, filename, n, provenance):
    path = directory / filename
    digest = sha256(path)
    if digest != record["files"].get(filename):
        raise ValueError(f"Changed prediction file: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.shape[1:] != (32, 512) or len(array) < n:
        raise ValueError(f"Prediction/data shape mismatch: {path}")
    provenance[str(path)] = digest
    provenance[str(directory / "run.json")] = sha256(directory / "run.json")
    return array


def checked_prediction(directory, method, seed, data_hashes, n, provenance):
    record = read_run(directory)
    contract = record["contract"]
    if (contract["method"] != method or contract["inference_seed"] != seed
            or contract["data_sha256"] != data_hashes
            or contract["count"] is not None
            or contract["layouts_sha256"] is not None):
        raise ValueError(f"Expected matching baseline data, method and seed: {directory}")
    array = checked_array(directory, record, "predictions.npy", n, provenance)
    if len(array) != contract["n_sections"]:
        raise ValueError(f"Prediction count differs from its run record: {directory}")
    return array, record["files"]["predictions.npy"]


def write_table(path, ratios):
    labels = dict(tabpfn="TabPFN", schemagan="SchemaGAN", kriging="Kriging",
                  nearest_neighbour="Nearest Neighbour")
    lines = ["# Table B.1: spatial difference ratios", ""]
    for metric_index, metric in enumerate(METRICS):
        lines += [f"## {metric.replace('_', ' ')}", "",
                  "| Method | " + " | ".join(f"{d}{lag}" for d, lag in LAGS) + " |",
                  "|---|" + "---:|" * len(LAGS)]
        for method in METHODS:
            values = np.median(ratios[method][:, :, metric_index], axis=0)
            lines.append(f"| {labels[method]} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
        lines.append("")
    lines += ["V: depth rows; H: grid columns. All in-grid pairs are included.",
              "These ratios do not measure boundary positions or layer connectivity.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Score a prefix for a small check")
    parser.add_argument("--block-size", type=int, default=16, help="Sections per block (1--64)")
    args = parser.parse_args()
    if not 1 <= args.block_size <= 64 or (args.limit is not None and args.limit < 1):
        parser.error("Use a positive limit and block size between 1 and 64")
    args.data = (ROOT / args.data).resolve()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    started = time.monotonic()
    _, truth, sections, manifest = load_data(args.data, args.limit)
    if manifest["config"]["split"] != "test" or manifest.get("normalized"):
        raise ValueError("Use synthetic test sections in original Ic units")
    n = len(truth)
    baseline = args.baseline.resolve()
    provenance = {str((args.data.resolve() / "manifest.json")): sha256(args.data / "manifest.json")}
    for filename, digest in manifest["output_sha256"].items():
        provenance[str(args.data.resolve() / filename)] = digest
    target = score_sections(truth, n, args.block_size)
    if np.any(target <= 0):
        raise ValueError("A target descriptor is zero; its ratio is undefined")
    ratios = {}
    for method in ("tabpfn", "kriging", "nearest_neighbour"):
        fields, _ = checked_prediction(baseline / method, method, 0,
                                       manifest["output_sha256"], n, provenance)
        ratios[method] = ratios_to_target(score_sections(fields, n, args.block_size), target)
        print(f"Scored {method}: {n} sections", flush=True)

    directory = baseline / "schemagan_summary"
    summary = read_run(directory)
    seeds = summary["seeds"]
    if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or summary["n_draws"] != len(seeds)
            or summary["data_sha256"] != manifest["output_sha256"]):
        raise ValueError("SchemaGAN aggregation seeds or data do not match")
    if "contract" in summary:
        contract = summary["contract"]
        if (contract["count"] is not None or contract["layouts_sha256"] is not None
                or contract["seeds"] != seeds or contract["data_sha256"] != manifest["output_sha256"]
                or contract["input_predictions_sha256"] != summary["input_predictions_sha256"]):
            raise ValueError("SchemaGAN summary is not a matching baseline aggregation")
    median_fields = checked_array(directory, summary, "medians.npy", n, provenance)
    if "contract" in summary and len(median_fields) != summary["contract"]["n_sections"]:
        raise ValueError("SchemaGAN summary count differs from its run record")
    cases = sorted(set((0, min(777, n - 1), n // 2, n - 1)))
    sampled_draws = []
    draw_ratios = np.zeros_like(target)
    for index, seed in enumerate(seeds):
        fields, digest = checked_prediction(baseline / "schemagan" / f"seed_{seed}",
                                            "schemagan", seed, manifest["output_sha256"], n, provenance)
        if (digest != summary["input_predictions_sha256"].get(str(seed))
                or len(fields) != len(median_fields)):
            raise ValueError("SchemaGAN summary was made from different draws")
        draw_ratios += ratios_to_target(score_sections(fields, n, args.block_size), target)
        sampled_draws.append(np.asarray(fields[cases], np.float64))
        if (index + 1) % 10 == 0 or index + 1 == len(seeds):
            print(f"Scored SchemaGAN draws: {index + 1}/{len(seeds)}", flush=True)
    ratios["schemagan"] = draw_ratios / len(seeds)
    np.testing.assert_array_equal(np.median(np.stack(sampled_draws), axis=0), median_fields[cases])
    median_ratios = ratios_to_target(score_sections(median_fields, n, args.block_size), target)
    comparisons, findings = compare_output_summaries(ratios["schemagan"], median_ratios, ratios["tabpfn"])

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "table_b1.csv", summaries(ratios))
    write_table(args.output / "table_b1.md", ratios)
    write_csv(args.output / "output_summary_comparison.csv", comparisons)
    write_json(args.output / "findings.json", findings)
    np.savez_compressed(args.output / "per_section.npz", target=target, **ratios,
                        schemagan_pixelwise_median=median_ratios,
                        lags=np.asarray([f"{d}{lag}" for d, lag in LAGS]),
                        descriptors=np.asarray(METRICS),
                        section_ids=np.asarray([r["section_id"] for r in sections]))
    write_json(args.output / "run.json", dict(
        status="completed", n_sections=n, n_draws=len(seeds), seeds=seeds,
        full_analysis=n == 4000 and seeds == list(range(100)),
        source_commit=SOURCE_COMMIT,
        numerical_environment=dict(python=platform.python_version(), numpy=np.__version__),
        lags=LAGS, descriptors=METRICS, block_size=args.block_size, threads=1,
        pairs="All in-grid pairs at each separation, including input pixels; equal weights",
        aggregation="Ratio within section; average SchemaGAN draws, then median across sections",
        no_distance_normalisation=True, no_inference_or_training=True,
        cached_pixelwise_median_reference_cases=cases, inputs_sha256=provenance,
        data_sha256=manifest["output_sha256"],
        script_sha256={p.name: sha256(p) for p in (
            Path(__file__), ROOT / "scripts/common.py", ROOT / "scripts/tabpfn_cpt_protocol.py")},
        files={p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
        elapsed_seconds=time.monotonic() - started,
    ))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
