"""Compare initial and fine-tuned predictions, including fixed common evaluation pixels."""

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
from common import load_data, error_metrics
from analyse_runs import verified


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-data", type=Path, required=True)
    p.add_argument("--fresh-data", type=Path, required=True)
    p.add_argument(
        "--baseline-runs",
        type=Path,
        required=True,
        help="Contains base, ft0 and ft1 runs",
    )
    p.add_argument(
        "--fresh-runs", type=Path, required=True, help="Contains base, ft0 and ft1 runs"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    baseline, truth, _, baseline_info = load_data(a.baseline_data)
    fresh, fresh_truth, _, fresh_info = load_data(a.fresh_data)
    np.testing.assert_array_equal(truth, fresh_truth)
    records = []
    for placement, source, directory in [
        ("baseline", baseline, a.baseline_runs),
        ("fresh", fresh, a.fresh_runs),
    ]:
        for model in ("base", "ft0", "ft1"):
            run = json.loads((directory / model / "run.json").read_text())
            expected = baseline_info if placement == "baseline" else fresh_info
            if run["contract"]["data_sha256"] != expected["output_sha256"]:
                raise ValueError(f"Prediction inputs do not match {placement} data")
            prediction = np.load(
                verified(directory / model, "predictions.npy"), mmap_mode="r"
            )
            if prediction.shape != truth.shape:
                raise ValueError("Prediction/data shape mismatch")
            for i in range(len(truth)):
                masks = [
                    ("all_pixels", np.ones((32, 512), bool)),
                    ("unobserved_pixels", source[i] == 0),
                    ("common_unobserved_pixels", (baseline[i] == 0) & (fresh[i] == 0)),
                ]
                for scope, mask in masks:
                    records.append(
                        dict(
                            array_index=i,
                            placement=placement,
                            model=model,
                            scope=scope,
                            **error_metrics(prediction[i], truth[i], mask),
                        )
                    )
    frame = pd.DataFrame(records)
    frame.to_csv(a.output / "by_section.csv", index=False)
    summary = frame.groupby(["placement", "scope", "model"])[
        ["mae", "mse", "rmse"]
    ].mean()
    summary.to_csv(a.output / "means.csv")
    records = []
    for placement in ("baseline", "fresh"):
        for scope in ("all_pixels", "unobserved_pixels", "common_unobserved_pixels"):
            for model in ("ft0", "ft1"):
                records.append(
                    dict(
                        placement=placement,
                        scope=scope,
                        model=model,
                        **{
                            m: 100
                            * (
                                1
                                - summary.loc[(placement, scope, model), m]
                                / summary.loc[(placement, scope, "base"), m]
                            )
                            for m in ("mae", "rmse")
                        },
                    )
                )
    pd.DataFrame(records).to_csv(a.output / "reductions_percent.csv", index=False)
    print(a.output)


if __name__ == "__main__":
    main()
