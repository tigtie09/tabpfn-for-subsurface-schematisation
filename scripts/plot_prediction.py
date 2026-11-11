"""Plot one computed synthetic prediction alongside its inputs and target section."""

import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common import ROOT, load_data, sha256

LABELS = {
    "tabpfn": "TabPFN predictive median",
    "schemagan": "SchemaGAN",
    "kriging": "Kriging",
    "nearest_neighbour": "Nearest neighbour",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/test")
    parser.add_argument(
        "--run", type=Path, required=True, help="Completed prediction run"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based index in lexicographic section order",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New PNG, PDF or SVG file"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new filename")
    if args.output.suffix.lower() not in (".png", ".pdf", ".svg"):
        parser.error("Output must have a .png, .pdf or .svg extension")
    source, truth, sections, manifest = load_data(args.data)
    report = json.loads((args.run / "run.json").read_text())
    contract = report.get("contract", {})
    method = contract.get("method")
    if method not in LABELS or report.get("status") != "completed":
        parser.error(
            "Use a completed single-model prediction run; for SchemaGAN, select a seed directory"
        )
    if contract.get("count") is not None:
        parser.error(
            "This small plotting example uses baseline or fresh inputs, not the CPT-count runs"
        )
    if contract.get("data_sha256") != manifest["output_sha256"]:
        parser.error("The prediction run and prepared data do not match")
    path = args.run / "predictions.npy"
    if sha256(path) != report["files"]["predictions.npy"]:
        parser.error("Prediction checksum differs from the completed run")
    predictions = np.load(path, mmap_mode="r", allow_pickle=False)
    if predictions.shape != (contract["n_sections"], 32, 512) or len(predictions) > len(
        truth
    ):
        parser.error("Prediction and data shapes do not match")
    if not 0 <= args.index < len(predictions):
        parser.error("Section index is outside this prediction run")
    observed, target, prediction = (
        source[args.index],
        truth[args.index],
        predictions[args.index],
    )
    if not np.isfinite(prediction).all():
        parser.error("Prediction contains non-finite values")
    vmin = min(float(target.min()), float(prediction.min()))
    vmax = max(float(target.max()), float(prediction.max()))
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    label = LABELS[method]
    if method == "schemagan":
        label += f" (inference seed {contract['inference_seed']})"
    fig, axes = plt.subplots(
        3, 1, figsize=(9, 5.5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, values, title in zip(
        axes,
        (np.ma.masked_equal(observed, 0), target, prediction),
        ("Input CPT observations", "Target section", label),
    ):
        image = ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(title=title, ylabel="Grid row")
    axes[-1].set_xlabel("Grid column")
    fig.suptitle(
        f"Synthetic section {sections[args.index]['section_id']} (array index {args.index})"
    )
    fig.colorbar(
        image,
        ax=axes.tolist(),
        label=r"Soil behaviour type index $I_\mathrm{c}$",
        shrink=0.9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
