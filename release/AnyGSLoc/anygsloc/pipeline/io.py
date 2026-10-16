"""Small, explicit IO helpers for reproducible runs."""

import json
import os
from pathlib import Path

from anygsloc.artifacts import sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def reference(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_config(path, asset_root=None, output=None):
    path = Path(path).resolve()
    config = read_json(path)
    assets = Path(
        asset_root or os.environ.get("ANYGSLOC_ASSETS", path.parent / "../assets")
    ).resolve()
    config["asset_root"] = str(assets)
    config["output"] = str(
        Path(output or path.parent / "../runs" / path.stem).resolve()
    )
    config["config_sha256"] = sha256(path)
    for name, scene in config["scenes"].items():
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid scene name")
        for key in (
            "dataset",
            "prior",
            "observation_cache",
            "selector_state",
            "full_map",
            "compact_map",
        ):
            if key in scene and isinstance(scene[key], str):
                scene[key] = str((assets / scene[key]).resolve())
        if scene.get("query_mask", "none") not in ("none", "explicit_valid_mask"):
            raise ValueError("unknown query mask policy")
    return config


def prepare_output(path, inputs):
    """Each stage has its own output directory; incomplete stages are never reused."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "inputs.json", inputs)
    return path


def complete(path, outputs, **metadata):
    path = Path(path)
    write_json(
        path / "complete.json",
        {"outputs": {name: reference(path / name) for name in outputs}, **metadata},
    )


def require_complete(path):
    path = Path(path)
    record = read_json(path / "complete.json")
    for name, expected in record["outputs"].items():
        if sha256(path / name) != expected["sha256"]:
            raise ValueError(f"stage output changed: {path / name}")
    return record
