"""Recreate the six-CPT benchmark using the raw manifest and configured order.

The sampling procedure follows fabcamo/schemaGAN, under the MIT license;
see THIRD_PARTY.md for the full licence.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_section(path):
    # groupby('z') in the original code sorts z and preserves order within z.
    frame = pd.read_csv(path, usecols=["x", "z", "IC"])
    frame = frame.sort_values("z", kind="stable")
    if len(frame) != 32 * 512:
        raise ValueError(f"Unexpected row count: {path}")
    if not (np.array_equal(frame.x.to_numpy(), np.tile(np.arange(512), 32))
            and np.array_equal(frame.z.to_numpy(), np.repeat(np.arange(32), 512))):
        raise ValueError(f"Unexpected coordinates or row order: {path}")
    section = frame.IC.to_numpy(dtype=np.float32).reshape(32, 512)
    if not np.isfinite(section).all() or not (section > 0).all():
        raise ValueError(f"Invalid IC values: {path}")
    return section


def sample_cpts(section, rng, miss_rate, min_distance):
    depth, width = section.shape
    n_keep = width - int(miss_rate * width)
    columns = []
    while len(columns) < n_keep:
        candidate = int(rng.uniform(0, width))
        if all(abs(candidate - existing) > min_distance for existing in columns):
            columns.append(candidate)
    source = np.zeros_like(section)
    # Consume one draw for EVERY column, including unobserved columns, as the
    # released code does. Sampling only the six kept columns changes later RNG.
    for x in range(width):
        removed = int(rng.triangular(0, 0, depth / 2))
        if x in columns:
            source[:depth - removed, x] = section[:depth - removed, x]
    return source


def load_sections(manifest_path, split, order):
    if order != "relative_path_lexicographic":
        raise ValueError(f"Unsupported section order: {order}")
    with manifest_path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    paths = [row["relative_path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Manifest contains duplicate section paths")
    for row in rows:
        path = Path(row["relative_path"])
        if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != split:
            raise ValueError(f"Unexpected raw section path: {path}")
        if path.name != f"cs_{int(row['section_id'])}.csv":
            raise ValueError(f"Section ID does not match filename: {path}")
    return sorted(rows, key=lambda row: row["relative_path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/synthetic_cpt/baseline6_test.json",
                        help="Preprocessing config (default: configs/synthetic_cpt/baseline6_test.json)")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory; existing directories are never overwritten")
    parser.add_argument("--limit", type=int, help="Prepare a prefix for a small check")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config["section_shape"] != [32, 512, 1] or config["dtype"] != "float32"
            or config["miss_rate"] != 0.99 or config["min_distance"] != 51
            or config["random_generator"] != "numpy.random.RandomState (MT19937)"
            or config["missing_value"] != 0.0):
        parser.error("This script implements the fixed six-CPT benchmark only")
    manifest_path = ROOT / config["section_manifest"]
    rows = load_sections(manifest_path, config["split"], config["section_order"])
    if len(rows) != config["n_sections"]:
        parser.error("Manifest section count does not match config")
    n = len(rows) if args.limit is None else args.limit
    if not 1 <= n <= len(rows):
        parser.error("--limit must be between 1 and the manifest section count")
    if n < len(rows) and args.output_dir is None:
        parser.error("Use --output-dir for a partial run")
    output = args.output_dir or ROOT / config["output_dir"]
    raw_dir = ROOT / config["raw_dir"]
    output.mkdir(parents=True, exist_ok=False)
    shape = (n, 32, 512, 1)
    source_path, target_path = output / "src_images.npy", output / "tar_images.npy"
    source = np.lib.format.open_memmap(source_path, mode="w+", dtype=np.float32, shape=shape)
    target = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float32, shape=shape)
    rng = np.random.RandomState(config["seed"])
    min_observed, max_observed = 32 * 512, 0
    for i, row in enumerate(rows[:n]):
        path = raw_dir / row["relative_path"]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"Raw CSV does not match manifest: {path}")
        section = read_section(path)
        cpts = sample_cpts(section, rng, config["miss_rate"], config["min_distance"])
        if (cpts != 0).any(axis=0).sum() != 6:
            raise ValueError(f"Unexpected CPT count at array index {i}")
        observed = int(np.count_nonzero(cpts))
        min_observed, max_observed = min(min_observed, observed), max(max_observed, observed)
        source[i, :, :, 0], target[i, :, :, 0] = cpts, section
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"Prepared {i + 1}/{n} sections", flush=True)
    source.flush()
    target.flush()
    del source, target
    hashes = {path.name: sha256(path) for path in (source_path, target_path)}
    reference_match = None
    if n == len(rows) and "reference_sha256" in config:
        reference_match = hashes == config["reference_sha256"]
    metadata = {
        "dataset_id": config["dataset_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_sections": n,
        "shape": list(shape),
        "dtype": "float32",
        "values": "IC in original units; unobserved source pixels are zero",
        "normalized": False,
        "config": config,
        "config_sha256": sha256(args.config),
        "section_manifest_sha256": sha256(manifest_path),
        "script_sha256": sha256(__file__),
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "observed_pixels_per_section": {"min": min_observed, "max": max_observed},
        "output_sha256": hashes,
        "full_reference_files_match": reference_match,
    }
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    if reference_match is False:
        raise ValueError("Generated file hashes differ from the historical benchmark")
    print(json.dumps({"output_dir": str(output), "sha256": hashes,
                      "full_reference_files_match": reference_match}, indent=2))


if __name__ == "__main__":
    main()
