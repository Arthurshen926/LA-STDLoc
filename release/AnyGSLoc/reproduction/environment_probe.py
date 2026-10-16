"""Run inside one selected environment; no training or model downloads."""

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    profile = json.loads((root / "environments/profiles.json").read_text())[
        args.profile
    ]
    expected = profile["packages"]
    if isinstance(expected, str):
        expected = json.loads((root / expected).read_text())
    errors, packages = [], {}
    for name, version in expected.items():
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
        if packages[name] != version:
            errors.append(f"{name}: expected {version}, found {packages[name]}")
    if platform.python_version_tuple()[:2] != tuple(profile["python"].split(".")):
        errors.append("Python major/minor differs from the accepted profile")
    imports = {}
    for name in profile["imports"]:
        try:
            importlib.import_module(name)
            imports[name] = "passed"
        except Exception as exc:
            imports[name] = str(exc)
            errors.append(f"cannot import {name}: {exc}")
    gpu = None
    try:
        import torch

        if torch.version.cuda != profile["cuda"]:
            errors.append(f"PyTorch CUDA differs: {torch.version.cuda}")
        if args.device != "cpu":
            device = torch.device(args.device)
            value = torch.ones(4, device=device).sum().item()
            assert value == 4
            gpu = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
    except Exception as exc:
        errors.append(f"runtime check: {exc}")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        candidate = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
        nvcc = str(candidate) if candidate.is_file() else None
    result = {
        "profile": args.profile,
        "conda_env": profile["conda_env"],
        "python": platform.python_version(),
        "packages": packages,
        "imports": imports,
        "device": args.device,
        "gpu": gpu,
        "nvcc_version": subprocess.check_output([nvcc, "--version"], text=True).strip()
        if nvcc
        else None,
        "validation_scope": profile["validation_scope"],
        "errors": errors,
        "passed": not errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
