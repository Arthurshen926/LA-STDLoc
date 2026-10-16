"""Frozen, resumable offline pipeline. Never launches a localization evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from anygsloc.artifacts import sha256


@dataclass(frozen=True)
class MappingConfig:
    keypoints: int = 2048
    render_alpha_minimum: float = 0.05
    render_neighborhood_radius: int = 4
    render_supersampling: int = 1
    nms_radius: int = 4
    audit_shards: int = 2
    cpu_threads: int = 4
    triangulation_workers: int = 4


MAPPING = MappingConfig()


@contextmanager
def exclusive_build(output):
    """Protect one output directory from simultaneous resume processes."""
    import fcntl

    with (Path(output) / ".build.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process owns this build directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_prior(args):
    if args.prior_manifest:
        p = Path(args.prior_manifest).resolve()
        data = json.loads(p.read_text())
        if (
            data.get("prior_kind") != "rgb_only"
            or data.get("prior_training_used_feature_loss") is not False
        ):
            raise ValueError("expected frozen RGB-only prior manifest")
        ply = Path(data["exported_ply"]).resolve()
        if sha256(ply) != data["exported_ply_sha256"]:
            raise ValueError("prior PLY hash mismatch")
        return dict(
            ply=str(ply),
            kind=data["gaussian_type"],
            sh_degree=int(data["sh_degree"]),
            white_background=bool(data.get("white_background", False)),
            manifest_sha256=sha256(p),
        )
    if args.gaussian_type is None or args.sh_degree is None:
        raise ValueError("explicit PLY requires --gaussian-type and --sh-degree")
    return dict(
        ply=str(Path(args.gaussian_ply).resolve()),
        kind=args.gaussian_type,
        sh_degree=args.sh_degree,
        white_background=args.white_background,
    )


def make_plan(args, prior):
    output = Path(args.output).resolve()
    observation = output / "observations/rendered_rgb_feature_cache.pt"
    full = output / "full"
    compact = output / "compact"
    audits = [
        output / "render_audit" / f"shard_{i}.pt" for i in range(MAPPING.audit_shards)
    ]
    shared = [
        "--dataset",
        str(Path(args.dataset).resolve()),
        "--images",
        args.images,
        "--gaussian-ply",
        prior["ply"],
        "--gaussian-type",
        prior["kind"],
        "--sh-degree",
        str(prior["sh_degree"]),
    ]
    if prior["white_background"]:
        shared += ["--white-background"]

    def stage(name, module, options, outputs):
        return dict(
            name=name,
            command=[sys.executable, "-m", module, *map(str, options)],
            outputs=list(map(str, outputs)),
        )

    stages = [
        stage(
            "render_observations",
            "anygsloc.mapping.render_observations",
            [*shared, "--output-dir", observation.parent],
            [observation],
        )
    ]
    for i, audit in enumerate(audits):
        stages.append(
            stage(
                f"render_audit_{i}",
                "anygsloc.mapping.audit_render",
                [
                    *shared,
                    "--observation-cache",
                    observation,
                    "--output",
                    audit,
                    "--shard-index",
                    i,
                    "--shard-count",
                    MAPPING.audit_shards,
                    "--device",
                    "cuda:0",
                ],
                [audit],
            )
        )
    stages.append(
        stage(
            "projective_map",
            "anygsloc.mapping.materialize",
            [
                "--observation-cache",
                observation,
                "--v2-audit-shards",
                *audits,
                "--output-dir",
                full,
                "--device",
                "cuda:0",
                "--cpu-threads",
                MAPPING.cpu_threads,
                "--triangulation-workers",
                MAPPING.triangulation_workers,
            ],
            [
                full / "projective_anchor_map.pt",
                full / "identity_metric.pt",
                full / "report.json",
            ],
        )
    )
    stages.append(
        stage(
            "compact_map",
            "anygsloc.cli",
            [
                "compact-map",
                "--map-dir",
                full,
                "--cache",
                observation,
                "--output",
                compact,
            ],
            [
                compact / "projective_anchor_map.pt",
                compact / "identity_metric.pt",
                compact / "selector_trace.pt",
                compact / "deployment_map.pt",
            ],
        )
    )
    return stages


def contract(args, prior, stages):
    dataset = Path(args.dataset).resolve()
    metadata = {}
    for relative in [
        "sparse/0/cameras.bin",
        "sparse/0/cameras.txt",
        "sparse/0/images.bin",
        "sparse/0/images.txt",
        "sparse/0/list_test.txt",
        "dataset_test.txt",
        "dataset_train.txt",
        "masks.pkl",
        f"{args.images}/masks.pkl",
    ]:
        p = dataset / relative
        if p.is_file():
            metadata[relative] = sha256(p)
    if not any(n.startswith("sparse/0/cameras.") for n in metadata):
        raise ValueError("missing COLMAP camera metadata")
    code_root = Path(__file__).resolve().parents[1]
    code = {
        str(p.relative_to(code_root)): sha256(p)
        for p in sorted(code_root.rglob("*.py"))
    }
    value = dict(
        schema="anygsloc_build_v1",
        prior=prior,
        prior_sha256=sha256(prior["ply"]),
        dataset=str(dataset),
        image_domain=args.image_domain,
        metadata=metadata,
        parameters=asdict(MAPPING),
        stages=stages,
        code=code,
    )
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value, digest


def run_plan(stages, output, digest, *, resume=False, env=None):
    """Never equate file existence with completion; preserve interrupted outputs."""
    output = Path(output)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        receipt = logs / f"{stage['name']}.json"
        targets = [Path(p) for p in stage["outputs"]]
        if resume and receipt.exists():
            old = json.loads(receipt.read_text())
            if old.get("status") == "complete" and old.get("contract_sha256") == digest:
                if all(
                    p.is_file() and old["outputs"].get(str(p)) == sha256(p)
                    for p in targets
                ):
                    continue
                raise ValueError(f"completed stage changed on disk: {stage['name']}")
        # Partial artifacts are evidence. Do not overwrite or call them done.
        if any(p.exists() for p in targets):
            raise FileExistsError(
                f"partial/unbound stage {stage['name']}; preserve outputs and use a fresh build directory"
            )
        started = time.perf_counter()
        record = dict(
            status="running", contract_sha256=digest, command=stage["command"]
        )
        atomic_json(receipt, record)
        try:
            with (logs / f"{stage['name']}.log").open("a") as log:
                subprocess.run(
                    stage["command"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not all(p.is_file() for p in targets):
                raise RuntimeError(
                    f"stage did not produce all outputs: {stage['name']}"
                )
            record.update(
                status="complete", outputs={str(p): sha256(p) for p in targets}
            )
        except BaseException as error:
            record.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                error=repr(error),
            )
            raise
        finally:
            record["wall_seconds"] = time.perf_counter() - started
            atomic_json(receipt, record)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--images", default="processed")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prior-manifest")
    source.add_argument("--gaussian-ply")
    p.add_argument("--gaussian-type", choices=["2dgs", "3dgs"])
    p.add_argument("--sh-degree", type=int)
    p.add_argument("--white-background", action="store_true")
    p.add_argument(
        "--image-domain",
        choices=["rectified"],
        required=True,
        help="explicit attestation: prior and COLMAP cameras share the correctly rectified domain",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    prior = resolve_prior(args)
    stages = make_plan(args, prior)
    value, digest = contract(args, prior, stages)
    if args.dry_run:
        print(json.dumps(dict(contract_sha256=digest, **value), indent=2))
        return
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_build(output):
        manifest = output / "build_manifest.json"
        if args.resume:
            if (
                not manifest.is_file()
                or json.loads(manifest.read_text())["contract_sha256"] != digest
            ):
                raise ValueError("resume contract differs; use a new output directory")
        else:
            atomic_json(manifest, dict(contract_sha256=digest, **value))
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }
        run_plan(stages, output, digest, resume=args.resume, env=env)


if __name__ == "__main__":
    main()
