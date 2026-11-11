"""Synthetic-only inputs and scoring shared by the public entry points."""

from pathlib import Path
import json
import numpy as np
from tabpfn_cpt_protocol import read_csv, sha256, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0.5, 0.8, 0.9, 0.95)
QUANTILES = (0.025, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.975)
COUNTS = (3, 6, 12, 24, 48)


def resolve(path):
    return ROOT / Path(path)


def load_data(directory, limit=None):
    directory = resolve(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Prepared data changed: {name}")
    config = manifest["config"]
    rows = sorted(
        (
            r
            for r in read_csv(ROOT / config["section_manifest"])
            if r["split"] == config["split"]
        ),
        key=lambda r: r["relative_path"],
    )
    source = np.load(directory / "src_images.npy", mmap_mode="r", allow_pickle=False)
    truth = np.load(directory / "tar_images.npy", mmap_mode="r", allow_pickle=False)
    n = len(source) if limit is None else limit
    if (
        source.shape != truth.shape
        or source.shape[1:] != (32, 512, 1)
        or not 1 <= n <= len(source)
    ):
        raise ValueError("Invalid prepared arrays or requested limit")
    return source[:n, :, :, 0], truth[:n, :, :, 0], rows[:n], manifest


def error_metrics(prediction, truth, mask):
    error = prediction[mask].astype(np.float64) - truth[mask].astype(np.float64)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Scoring requires finite predictions and a nonempty mask")
    mse = float(np.mean(error**2))
    return dict(
        n_pixels=len(error), mae=float(np.mean(np.abs(error))), mse=mse, rmse=mse**0.5
    )


def observation_mask(columns, depths, count):
    mask = np.zeros((32, 512), bool)
    mask[:, columns[:count]] = np.arange(32)[:, None] < depths[None, :count]
    return mask


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)
