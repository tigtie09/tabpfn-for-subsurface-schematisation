"""Run a declared baseline/count experiment sequentially in pinned model environments."""

import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    command = ["uv", "run", "--locked", str(ROOT / "scripts" / script), *map(str, args)]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=("baseline", "counts"))
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(100)))
    p.add_argument(
        "--methods",
        nargs="+",
        choices=("tabpfn", "schemagan", "kriging", "nearest_neighbour"),
        default=["tabpfn", "schemagan", "kriging", "nearest_neighbour"],
    )
    p.add_argument("--layouts", type=Path, default=ROOT / "data/processed/counts")
    a = p.parse_args()
    extra = ["--data", a.data.resolve()] + (["--limit", a.limit] if a.limit else [])
    counts = (3, 6, 12, 24, 48) if a.experiment == "counts" else (None,)
    for count in counts:
        folder = a.output.resolve() / (f"n_{count}" if count else "baseline")
        conditions = (
            ["--layouts", a.layouts.resolve(), "--count", count] if count else []
        )
        for method in a.methods:
            if count and method not in ("tabpfn", "schemagan"):
                continue
            if method == "schemagan":
                for seed in a.seeds:
                    run(
                        "predict_schemagan.py",
                        *extra,
                        *conditions,
                        "--seed",
                        seed,
                        "--output",
                        folder / method / f"seed_{seed}",
                    )
                run(
                    "aggregate_schemagan.py",
                    *extra,
                    *conditions,
                    "--draws",
                    folder / method,
                    "--output",
                    folder / "schemagan_summary",
                    "--seeds",
                    *a.seeds,
                    *(["--intervals"] if count is None else []),
                )
            else:
                intervals = (
                    ["--intervals"]
                    if count is None and method in ("tabpfn", "kriging")
                    else []
                )
                run(
                    f"predict_{method}.py",
                    *extra,
                    *conditions,
                    *intervals,
                    "--output",
                    folder / method,
                )


if __name__ == "__main__":
    main()
