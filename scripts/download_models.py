"""Download a published checkpoint and verify the exact manuscript checksum."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=("schemagan", "tabpfn"))
    a = p.parse_args()
    if a.model == "schemagan":
        filename = "schemaGAN.h5"
        url = "https://zenodo.org/records/13143431/files/schemaGAN.h5?download=1"
        expected = "b4834d8749a738538fd02b494dd4951197f8ee7f47f95f2be0add53bc6e5a987"
    else:
        settings = json.loads((ROOT / "configs/tabpfn.json").read_text())["model"]
        filename = Path(settings["path"]).name
        expected = settings["sha256"]
        url = f"https://huggingface.co/Prior-Labs/tabpfn_3/resolve/{settings['revision']}/{filename}"
        print(
            "TabPFN weights and outputs use the upstream non-commercial licence; see THIRD_PARTY.md.",
            flush=True,
        )
    output = ROOT / "models" / filename
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        if sha(output) != expected:
            raise ValueError("Existing checkpoint has a different checksum")
        print("Verified", output)
        return
    part = output.with_suffix(output.suffix + ".part")
    if part.exists():
        raise FileExistsError(
            f"Incomplete download: {part}; rename or remove this partial file before retrying"
        )
    with urllib.request.urlopen(url, timeout=120) as response, part.open("xb") as f:
        shutil.copyfileobj(response, f, 1024 * 1024)
    if sha(part) != expected:
        raise ValueError("Downloaded checkpoint differs from the manuscript version")
    part.rename(output)
    print("Verified", output)


if __name__ == "__main__":
    main()
