"""Obtain the original Zenodo archive, or install the two bundled demo sections."""

import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/13143431/files/data.zip?download=1"
MD5 = "37c2454fdcdb160a284fe15ff45ead54"


def digest(path, kind="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, kind).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", help="Install two CSVs locally; no download"
    )
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "data/downloads/data.zip"
    )
    parser.add_argument(
        "--split", choices=("all", "train", "vali", "test"), default="all"
    )
    args = parser.parse_args()
    with (ROOT / "configs/synthetic_cpt/raw_sections.csv").open() as stream:
        records = list(csv.DictReader(stream))
    if args.demo:
        records = sorted(
            (r for r in records if r["split"] == "test"),
            key=lambda r: r["relative_path"],
        )[:2]
    elif args.split != "all":
        records = [r for r in records if r["split"] == args.split]
    if not args.demo:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        if not args.archive.exists():
            part = args.archive.with_suffix(".zip.part")
            if part.exists():
                raise FileExistsError(
                    f"Incomplete download: {part}; rename or remove this partial file before retrying"
                )
            print("Downloading the 5.19 GB original archive", flush=True)
            with (
                urllib.request.urlopen(URL, timeout=120) as response,
                part.open("xb") as output,
            ):
                shutil.copyfileobj(response, output, 1024 * 1024)
            if digest(part, "md5") != MD5:
                raise ValueError("Archive checksum differs from Zenodo")
            part.rename(args.archive)
        if digest(args.archive, "md5") != MD5:
            raise ValueError("Archive checksum differs from Zenodo")
        archive = zipfile.ZipFile(args.archive)
        members = {}
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if (
                len(parts) >= 2
                and parts[-2] in ("train", "vali", "test")
                and parts[-1].endswith(".csv")
            ):
                relative = "/".join(parts[-2:])
                if relative in members:
                    raise ValueError(f"Ambiguous archive member: {relative}")
                members[relative] = name
    for row in records:
        destination = ROOT / "data/raw" / row["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != row["sha256"]:
                raise ValueError(f"Existing CSV differs: {destination}")
            continue
        source = (
            (ROOT / "examples" / row["relative_path"]).open("rb")
            if args.demo
            else archive.open(members[row["relative_path"]])
        )
        with source, destination.open("xb") as output:
            shutil.copyfileobj(source, output)
        if (
            destination.stat().st_size != int(row["bytes"])
            or digest(destination) != row["sha256"]
        ):
            raise ValueError(f"CSV verification failed: {destination}")
    if not args.demo:
        archive.close()
    print(f"Verified {len(records)} sections in data/raw/")


if __name__ == "__main__":
    main()
