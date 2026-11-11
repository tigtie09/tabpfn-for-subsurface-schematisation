"""Create the nested CPT sequence or the non-overlapping fresh test inputs."""

import argparse
import os
from pathlib import Path

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"

import numpy as np
from common import ROOT, load_data, observation_mask, sha256, write_csv, write_json
from prepare_synthetic_cpt import sample_cpts


def nested(source):
    # Layout 0 of the original implementation; only this layout was adopted.
    rng = np.random.RandomState(20232023)
    columns = np.empty((len(source), 48), np.int16)
    depths = np.empty_like(columns, dtype=np.uint8)
    for i, image in enumerate(source):
        selected = np.flatnonzero((image != 0).any(axis=0))
        heights = (image != 0).sum(axis=0)
        selected = rng.permutation(selected)
        added = rng.choice(np.setdiff1d(np.arange(512), selected), 42, replace=False)
        added_heights = 32 - rng.triangular(0, 0, 16, 42).astype(int)
        columns[i] = np.r_[selected, added]
        depths[i] = np.r_[heights[selected], added_heights]
        np.testing.assert_array_equal(
            observation_mask(columns[i], depths[i], 6), image != 0
        )
    return columns, depths


def signature(source):
    lengths = (source != 0).sum(axis=0)
    cols = np.flatnonzero(lengths)
    return tuple(zip(cols.tolist(), lengths[cols].tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("counts", "fresh"))
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, truth, rows, manifest = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.kind == "counts":
        columns, depths = nested(source)
        np.savez_compressed(args.output / "layouts.npz", columns=columns, depths=depths)
        write_csv(
            args.output / "placements.csv",
            [
                dict(array_index=i, order=j, column=int(x), depth=int(z))
                for i, (xs, zs) in enumerate(zip(columns, depths))
                for j, (x, z) in enumerate(zip(xs, zs))
            ],
        )
        write_json(
            args.output / "manifest.json",
            dict(
                n_sections=len(source),
                layout=0,
                seed=20232023,
                target="complement of this section's 48-column mask",
                source_sha256=manifest["output_sha256"]["src_images.npy"],
                layouts_sha256=sha256(args.output / "layouts.npz"),
            ),
        )
    else:
        # Baseline masks are identical at corresponding split positions. Build
        # all 16,000 historical masks without loading training targets.
        old_rng = np.random.RandomState(20232023)
        dummy = np.ones((32, 512), np.float32)
        old = {signature(sample_cpts(dummy, old_rng, 0.99, 51)) for _ in range(16000)}
        rng = np.random.RandomState(20261022)
        fresh = np.stack([sample_cpts(t, rng, 0.99, 51) for t in truth])
        signatures = [signature(s) for s in fresh]
        if any(s in old for s in signatures):
            raise ValueError("A fresh placement repeats a historical placement")
        np.save(args.output / "src_images.npy", fresh[..., None])
        np.save(args.output / "tar_images.npy", truth[..., None])
        config = dict(
            manifest["config"], seed=20261022, dataset_id="synthetic_fresh_test"
        )
        write_json(
            args.output / "manifest.json",
            dict(
                config=config,
                n_sections=len(source),
                original_test_data=str(args.data),
                overlaps_with_training_validation_baseline_test=0,
                output_sha256={
                    name: sha256(args.output / name)
                    for name in ("src_images.npy", "tar_images.npy")
                },
            ),
        )
    print(args.output)


if __name__ == "__main__":
    main()
