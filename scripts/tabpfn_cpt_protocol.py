"""Grid coordinates, context extraction and file utilities from the manuscript implementation."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_arrays(source, mask):
    if (
        source.shape != (32, 512)
        or mask.shape != source.shape
        or mask.dtype != np.bool_
        or not mask.any()
    ):
        raise ValueError("Need a 32x512 source and a nonempty boolean observation mask")
    z, x = np.nonzero(mask)
    X = np.column_stack((x, z)).astype(np.float32)
    y = np.asarray(source[mask], dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Observed IC values must be finite")
    return X, y


def query_grid():
    z, x = np.indices((32, 512))
    return np.column_stack((x.ravel(), z.ravel())).astype(np.float32)
