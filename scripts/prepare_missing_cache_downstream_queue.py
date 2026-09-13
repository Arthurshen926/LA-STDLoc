"""Register downstream audits, map rebuilds, and base evaluations for recovered Cambridge caches.

The rendered observation caches are already complete.  This queue only audits those
immutable caches, rebuilds the feedback-free projective maps, and evaluates the
resulting maps.  It never overwrites the historical V24 tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POOL = Path("/mnt/pool/sqy")
RECOVERY = POOL / "anygsloc_cache_recovery_v2_20260912"
OUT = POOL / "anygsloc_missing_cache_downstream_20260912"
CODE = RECOVERY / "code"
PYTHON = "/root/miniconda3/envs/g4splat/bin/python"
DATASET = POOL / "Cambridge_stdloc"
PRIOR_ROOT = POOL / "stdloc_lafgs_v1_frozen_multiscene_20260731"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _env(gpu: int) -> dict[str, str]:
    return {
        "PYTHONPATH": str(CODE),
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "LD_LIBRARY_PATH": "/root/miniconda3/envs/g4splat/lib",
        "PATH": "/root/miniconda3/envs/g4splat/bin:/usr/local/cuda/bin:/usr/bin:/bin",
    }


def _audit_job(scene: str, shard: int, gpu: int) -> dict:
    cache = RECOVERY / scene / "observations" / "rendered_rgb_feature_cache.pt"
    ply = PRIOR_ROOT / scene / "prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply"
    output = RECOVERY / scene / "v2_audit" / f"render_quality_shard_{shard:03d}_of_002.pt"
    command = [
        PYTHON, "-u", "-m", "scripts.audit_v7_mapping_render_quality_shard",
        "--dataset", str(DATASET / scene), "--images", "processed",
        "--gaussian-ply", str(ply), "--gaussian-type", "2dgs", "--sh-degree", "3",
        "--observation-cache", str(cache), "--output", str(output),
        "--shard-index", str(shard), "--shard-count", "2", "--device", "cuda:0",
    ]
    return {
        "id": f"{scene}_audit_shard{shard}", "command": command,
        "cwd": str(CODE), "env": _env(gpu), "deps": [],
        "outputs": [str(output)],
    }


def _map_job(scene: str, gpu: int, audit_ids: list[str]) -> dict:
    cache = RECOVERY / scene / "observations" / "rendered_rgb_feature_cache.pt"
    ply = PRIOR_ROOT / scene / "prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply"
    audit = [RECOVERY / scene / "v2_audit" / f"render_quality_shard_{i:03d}_of_002.pt" for i in range(2)]
    output = OUT / scene / "projective_map"
    command = [
        PYTHON, "-u", "-m", "scripts.materialize_v8_v2_projective_map",
        "--observation-cache", str(cache), "--v2-audit-shards", *map(str, audit),
        "--output-dir", str(output), "--device", "cuda:0", "--cpu-threads", "4",
        "--triangulation-workers", "4",
    ]
    # Cross-lane completion is enforced by the wait jobs in each map lane.
    deps = [f"wait_{scene}_{i}" for i in range(len(audit_ids))]
    return {
        "id": f"{scene}_map", "command": command, "cwd": str(CODE),
        "env": _env(gpu), "deps": deps,
        "outputs": [str(output / "projective_anchor_map.pt"), str(output / "identity_metric.pt"), str(output / "report.json")],
    }


def _eval_job(scene: str, gpu: int, map_id: str) -> dict:
    output = OUT / scene / "evaluation" / "base_seed2026"
    map_dir = OUT / scene / "projective_map"
    command = [
        PYTHON, "-u", "-m", "scripts.evaluate", "--dataset", str(DATASET / scene),
        "--images", "processed", "--map", str(map_dir / "projective_anchor_map.pt"),
        "--metric-state", str(map_dir / "identity_metric.pt"), "--output", str(output),
        "--config", str(Path("/root/STDLoc/configs/anygsloc_mainline.yaml")),
        "--split", "test", "--device", "cuda:0", "--seed", "2026", "--deployment-mode",
    ]
    return {
        "id": f"{scene}_eval", "command": command, "cwd": str(CODE),
        "env": _env(gpu), "deps": [map_id],
        "outputs": [str(output / "summary.json"), str(output / "results.json")],
    }


def _write_lane(path: Path, jobs: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    (path / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    scenes = ["GreatCourt", "StMarysChurch"]
    for scene in scenes:
        for required in [
            RECOVERY / scene / "observations/rendered_rgb_feature_cache.pt",
            PRIOR_ROOT / scene / "prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply",
        ]:
            if not required.is_file():
                raise FileNotFoundError(required)

    # One GPU lane per GreatCourt audit shard; St Mary is serialized on GPU 2.
    _write_lane(root / "great_gpu0", [_audit_job("GreatCourt", 0, 0)])
    _write_lane(root / "great_gpu1", [_audit_job("GreatCourt", 1, 1)])
    _write_lane(root / "stmary_gpu2", [_audit_job("StMarysChurch", 0, 2), _audit_job("StMarysChurch", 1, 2)])

    # Map/evaluation lanes wait on state files from the audit lanes.  The wait
    # jobs do not touch CUDA and therefore leave all three GPUs available.
    for scene, gpu, audit_lanes in [
        ("GreatCourt", 0, [root / "great_gpu0", root / "great_gpu1"]),
        ("StMarysChurch", 1, [root / "stmary_gpu2"]),
    ]:
        jobs = []
        wait_ids = []
        for index, lane in enumerate(audit_lanes):
            wait_id = f"wait_{scene}_{index}"
            wait_ids.append(wait_id)
            jobs.append({
                "id": wait_id,
                "command": [PYTHON, "-u", "-m", "scripts.wait_queue_lane_finished", "--state", str(lane / "state.json")],
                "cwd": str(CODE), "env": _env(gpu), "deps": [], "outputs": [],
            })
        map_job = _map_job(scene, gpu, wait_ids)
        jobs.append(map_job)
        jobs.append(_eval_job(scene, gpu, map_job["id"]))
        _write_lane(root / f"{scene.lower()}_map_gpu{gpu}", jobs)

    protocol = {
        "schema": "anygsloc_recovered_cache_downstream_queue",
        "scope": "fresh V2 render audits, feedback-free projective-map rebuild, and one base test evaluation",
        "scenes": scenes,
        "recovery_root": str(RECOVERY),
        "prior_root": str(PRIOR_ROOT),
        "overwrite_original": False,
        "cache_sha256": {
            scene: _sha256(RECOVERY / scene / "observations/rendered_rgb_feature_cache.pt")
            for scene in scenes
        },
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"registered downstream lanes under {root}")


if __name__ == "__main__":
    main()
