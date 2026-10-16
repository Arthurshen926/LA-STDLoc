"""Recompose qualitative paper panels from hash-verified display assets."""

import argparse
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("figure", choices=["e1", "e3", "e4"])
    p.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get("ANYGSLOC_ASSETS", "assets")),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.asset_root / "figure_inputs" / args.figure
    expected = json.loads((source / "files.json").read_text())
    for relative, digest in expected.items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("display input changed: " + relative)
    shutil.copytree(source, args.output)
    output = args.output.resolve()

    def paths(value):
        if isinstance(value, dict):
            return {paths(k): paths(v) for k, v in value.items()}
        if isinstance(value, list):
            return [paths(v) for v in value]
        if isinstance(value, str) and value.startswith(("assets/", "native/")):
            return str(output / value)
        return value

    for filename in ["source_provenance.json", "E4_gaussian_prior.provenance.json"]:
        path = output / filename
        if path.exists():
            path.write_text(
                json.dumps(paths(json.loads(path.read_text())), indent=2) + "\n"
            )
    module = importlib.import_module("reproduction.figures." + args.figure)
    if args.figure == "e3":
        module.draw(json.loads((output / "source_provenance.json").read_text()), output)
    else:
        module.draw(output)
    captions = json.loads((Path(__file__).parent / "captions.json").read_text())
    stem = {
        "e1": "E1_final_localization",
        "e3": "E3_compact_map",
        "e4": "E4_gaussian_prior",
    }[args.figure]
    caption = captions[args.figure]
    (output / f"{stem}.caption.tex").write_text(caption + "\n")
    (output / f"{stem}.caption.txt").write_text(caption + "\n")
    path = output / f"{stem}.provenance.json"
    record = json.loads(path.read_text())
    record["caption"] = caption
    path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
