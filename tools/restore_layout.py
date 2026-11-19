"""Verify archived code and copy it into its original relative directory layout."""
import argparse
import csv
import hashlib
from pathlib import Path
import shutil


def restore(destination):
    root = Path(__file__).resolve().parents[1]
    destination = destination.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Choose a destination outside this repository")
    if destination.exists():
        raise FileExistsError("Destination must not already exist")
    with (root / "CODE_MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    copies = []
    for row in rows:
        source = (root / row["repository_path"]).resolve()
        target = (destination / row["source_path"]).resolve()
        if root not in source.parents or destination not in target.parents:
            raise ValueError("Manifest path escapes its root")
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {row['repository_path']}")
        copies.append((source, target))
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    reference = "tmp/phewas_260923/charlson_reference/LICENSE.md"
    shutil.copy2(root / "phase_4_reporting" / reference, destination / reference)
    print(f"Verified and restored {len(copies)} code files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New directory outside this repository")
    restore(parser.parse_args().destination)
