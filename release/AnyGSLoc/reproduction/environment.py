"""Select, check, and snapshot the paper's separate Conda environments."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "reproduction/environments/profiles.json"


def conda_executable(explicit=None):
    candidates = [
        explicit,
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path(sys.prefix) / "bin/conda"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return str(Path(path).resolve())
    raise FileNotFoundError(
        "Conda not found; use --conda /path/to/conda or --prefix /path/to/env"
    )


def environment_prefix(args, profile):
    if args.prefix:
        prefix = args.prefix.expanduser().resolve()
    else:
        conda = conda_executable(args.conda)
        inventory = json.loads(
            subprocess.check_output([conda, "env", "list", "--json"], text=True)
        )
        name = args.env or profile["conda_env"]
        matches = [Path(p) for p in inventory["envs"] if Path(p).name == name]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one environment named {name!r}; use --prefix to select explicitly"
            )
        prefix = matches[0]
    if not (prefix / "bin/python").is_file():
        raise FileNotFoundError(prefix / "bin/python")
    return prefix


def child_environment(prefix, profile):
    env = dict(os.environ)
    for key, first in [
        ("PATH", prefix / "bin"),
        ("LD_LIBRARY_PATH", prefix / "lib"),
        ("PYTHONPATH", ROOT),
    ]:
        env[key] = str(first) + (os.pathsep + env[key] if env.get(key) else "")
    env.update(CONDA_PREFIX=str(prefix), PYTHONNOUSERSITE="1")
    env.update(
        {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    )
    env.setdefault("MAX_JOBS", "4")
    for key, value in profile.get("environment", {}).items():
        env.setdefault(key, value)
    weights = (
        Path(env.get("ANYGSLOC_ASSETS", ROOT / "assets")) / "weights/superpoint_v1.pth"
    )
    if weights.is_file():
        env.setdefault("ANYGSLOC_SUPERPOINT_WEIGHTS", str(weights.resolve()))
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("list")
    for action in ["check", "run", "snapshot"]:
        sub = commands.add_parser(action)
        sub.add_argument("--profile", required=True)
        sub.add_argument("--prefix", type=Path)
        sub.add_argument(
            "--env", help="Override the profile's existing Conda environment name"
        )
        sub.add_argument("--conda", help="Conda executable when it is not on PATH")
        if action == "check":
            sub.add_argument(
                "--device",
                default="cuda:0",
                help="Use cpu for an import/version-only check",
            )
            sub.add_argument("--output", type=Path)
        elif action == "run":
            sub.add_argument(
                "--cwd",
                type=Path,
                default=ROOT,
                help="Producer repository directory for upstream commands",
            )
            sub.add_argument("python_args", nargs=argparse.REMAINDER)
        else:
            sub.add_argument(
                "--output",
                type=Path,
                required=True,
                help="Fresh private snapshot directory",
            )
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text())
    if args.action == "list":
        print(json.dumps(profiles, indent=2))
        return
    if args.profile not in profiles:
        parser.error("unknown profile; choose " + ", ".join(profiles))
    profile = profiles[args.profile]
    prefix = environment_prefix(args, profile)
    env = child_environment(prefix, profile)
    python = str(prefix / "bin/python")
    if args.action == "run":
        argv = args.python_args
        if argv[:1] == ["--"]:
            argv = argv[1:]
        if not argv:
            parser.error(
                "supply Python arguments after --, for example -- -m reproduction.baselines --help"
            )
        raise SystemExit(subprocess.call([python, *argv], env=env, cwd=args.cwd))
    if args.action == "check":
        command = [
            python,
            str(ROOT / "reproduction/environment_probe.py"),
            "--profile",
            args.profile,
            "--device",
            args.device,
        ]
        if args.output:
            command += ["--output", str(args.output.resolve())]
        raise SystemExit(subprocess.call(command, env=env, cwd=ROOT))
    conda = conda_executable(args.conda)
    args.output.mkdir(parents=True, exist_ok=False)
    exports = {
        "conda-explicit.txt": [conda, "list", "--prefix", str(prefix), "--explicit"],
        "environment.yml": [
            conda,
            "env",
            "export",
            "--prefix",
            str(prefix),
            "--no-builds",
        ],
        "pip-freeze.txt": [python, "-m", "pip", "freeze", "--all"],
    }
    for filename, command in exports.items():
        with (args.output / filename).open("x") as stream:
            subprocess.run(command, stdout=stream, env=env, check=True)
    (args.output / "snapshot.json").write_text(
        json.dumps(
            {
                "profile": args.profile,
                "prefix": str(prefix),
                "scope": "local restoration evidence; editable sources and compiled extensions are not bundled",
                "environment": {
                    key: env.get(key)
                    for key in [
                        "CC",
                        "CXX",
                        "CUDA_HOME",
                        "MAX_JOBS",
                        "TORCH_CUDA_ARCH_LIST",
                        "ULFLOC_FUSION_BATCH_SIZE",
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
