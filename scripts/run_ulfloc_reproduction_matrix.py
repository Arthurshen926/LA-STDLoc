#!/usr/bin/env python3
"""Resumable ULF-Loc reproduction matrix.

This runner deliberately keeps the released ULF-Loc tree separate from the
STDLoc tree.  It stages read-only dataset views, trains one ULF-Loc model per
scene, evaluates the official sparse+sparse->dense pipeline once, and writes
separate sparse and dense metric records from the official ``summary.json``
and ``results.json``.

The coordinator is safe to stop and restart.  A scene is considered complete
only after its final marker has been atomically written.  The same command
also acts as the per-scene worker; this avoids a second bespoke shell script
and makes all command lines explicit in the logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCENES: List[Dict[str, str]] = []
for scene in ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"):
    SCENES.append({"dataset": "7scenes", "scene": scene, "slug": scene})
for slug in (
    "apt1_kitchen",
    "apt1_living",
    "apt2_bed",
    "apt2_kitchen",
    "apt2_living",
    "apt2_luke",
    "office1_gates362",
    "office1_gates381",
    "office1_lounge",
    "office1_manolis",
    "office2_5a",
    "office2_5b",
):
    SCENES.append({"dataset": "12scenes", "scene": slug, "slug": slug})
for scene in ("GreatCourt", "KingsCollege", "OldHospital", "StMarysChurch", "ShopFacade"):
    SCENES.append({"dataset": "cambridge", "scene": scene, "slug": scene})

SCENE_BY_ID = {f"{item['dataset']}/{item['slug']}": item for item in SCENES}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def symlink_checked(dst: Path, src: Path) -> None:
    """Create a link without silently replacing an unrelated file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        raise RuntimeError(f"staging link mismatch: {dst} -> {os.readlink(dst)} (wanted {src})")
    if dst.exists():
        raise RuntimeError(f"staging path already exists and is not a matching link: {dst}")
    os.symlink(src, dst, target_is_directory=src.is_dir())


def materialize_torch_masks(src: Path, dst: Path) -> Dict[str, Any]:
    """Convert release mask pickle arrays to the tensor contract ULF expects.

    The prepared indoor references contain semantically identical NumPy mask
    arrays, while the released ULF-Loc training loop calls ``.cuda()`` on each
    mask.  Keep the source pickle untouched and write a stage-owned pickle of
    bool Torch tensors.  ``pickle.load`` remains the release code's loader, so
    this is only a representation adaptation, not a ULF source patch.
    """
    with src.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"invalid ULF mask mapping: {src}")
    converted: Dict[str, Any] = {}
    tensor_cache: Dict[int, Any] = {}
    shapes = set()
    # Import lazily so staging metadata-only commands do not require Torch.
    import torch

    for name, values in raw.items():
        if not isinstance(name, str) or not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError(f"invalid mask entry for {name!r} in {src}")
        tensors = []
        for value in values:
            # The released mask pickle intentionally shares identical arrays
            # across many image keys.  Preserve that sharing instead of
            # cloning every entry (which would expand a ~1 MB pickle to many
            # gigabytes during staging).
            cache_key = id(value)
            tensor = tensor_cache.get(cache_key)
            if tensor is None:
                tensor = torch.as_tensor(value)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                if tensor.dtype != torch.bool:
                    tensor = tensor.to(dtype=torch.bool)
                tensor_cache[cache_key] = tensor
            if tensor.ndim != 2:
                raise ValueError(f"mask {name!r} has shape {tuple(tensor.shape)}")
            shapes.add(tuple(tensor.shape))
            tensors.append(tensor)
        converted[name] = tuple(tensors)
    if len(shapes) != 1:
        raise ValueError(f"mask shapes are inconsistent in {src}: {sorted(shapes)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with tmp.open("wb") as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)
    return {
        "source": str(src),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "staged": str(dst),
        "entries": len(converted),
        "shape": list(next(iter(shapes))),
        "dtype": "torch.bool",
    }


def source_path(item: Mapping[str, str], data7: Path, data12: Path, cambridge: Path) -> Path:
    if item["dataset"] == "7scenes":
        return data7 / item["scene"]
    if item["dataset"] == "12scenes":
        return data12 / item["scene"]
    return cambridge / item["scene"]


def make_staged_scene(
    item: Mapping[str, str],
    stage_root: Path,
    data7: Path,
    data12: Path,
    cambridge: Path,
) -> Path:
    """Build a tiny, lower-case ULF-compatible view of a prepared scene.

    ULF-Loc's released reader dispatches on the path spelling (``7scenes``,
    ``12scenes``, ``cambridge``) and expects ``sfm_gt`` for indoor scenes.
    The prepared local datasets use ``sparse/0`` and ``processed`` instead.
    We therefore link metadata and images into a private staging directory;
    no source image or source reconstruction is modified.
    """
    src = source_path(item, data7, data12, cambridge)
    if not src.is_dir():
        raise FileNotFoundError(f"missing source scene: {src}")
    stage = stage_root / item["dataset"] / item["slug"]
    stage.mkdir(parents=True, exist_ok=True)

    if item["dataset"] in {"7scenes", "12scenes"}:
        src_sfm = src / "sparse" / "0"
        sfm_name = "sfm_gt"
    else:
        src_sfm = src / "sparse"
        sfm_name = "sparse"
    if not src_sfm.is_dir():
        raise FileNotFoundError(f"missing COLMAP metadata: {src_sfm}")

    # Copy the metadata directory as links so PLY conversion, if needed, is
    # written in staging rather than into the prepared source tree.
    sfm = stage / sfm_name
    sfm.mkdir(parents=True, exist_ok=True)
    for child in src_sfm.iterdir():
        symlink_checked(sfm / child.name, child)

    processed = src / "processed"
    if not processed.is_dir():
        raise FileNotFoundError(f"missing processed image tree: {processed}")
    images = stage / "images"
    images.mkdir(parents=True, exist_ok=True)
    for child in processed.iterdir():
        symlink_checked(images / child.name, child)
    if item["dataset"] in {"7scenes", "12scenes"}:
        mask = src / "masks.pkl"
        if mask.is_file():
            staged_mask = images / "masks.pkl"
            # Older interrupted attempts may have left the source symlink.
            # Replace only that known link; never overwrite an unrelated file.
            if staged_mask.is_symlink():
                if Path(os.readlink(staged_mask)) != mask:
                    raise RuntimeError(f"staging mask link mismatch: {staged_mask}")
                staged_mask.unlink()
            elif staged_mask.exists():
                raise RuntimeError(f"staging mask path already exists: {staged_mask}")
            mask_info = materialize_torch_masks(mask, staged_mask)
        else:
            mask_info = None
    else:
        mask_info = None

    if item["dataset"] == "cambridge":
        for name in ("dataset_train.txt", "dataset_test.txt"):
            file = src / name
            if file.is_file():
                symlink_checked(stage / name, file)

    atomic_json(
        stage / "ulfloc_stage_manifest.json",
        {
            "schema": "ulfloc_reproduction_stage_v1",
            "dataset": item["dataset"],
            "scene": item["scene"],
            "source_scene": str(src),
            "source_sfm": str(src_sfm),
            "source_images": str(processed),
            "sfm_layout": sfm_name,
            "image_layout": "processed_link_tree",
            "mask": mask_info,
        },
    )
    return stage


def config_for(item: Mapping[str, str], repo: Path) -> Path:
    if item["dataset"] == "7scenes":
        return repo / "configs" / "ulfloc_7scenes.yaml"
    if item["dataset"] == "12scenes":
        return repo / "configs" / "ulfloc_12scenes.yaml"
    return repo / "configs" / "ulfloc_cambridge.yaml"


def model_dir(scene_root: Path) -> Path:
    return scene_root / "model"


def training_complete(model: Path) -> bool:
    cloud = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    test = model / "test"
    return cloud.is_file() and (test / "keypoints_sampled_idx.pkl").is_file() and (
        test / "keypoints_features.pkl"
    ).is_file()


def evaluation_complete(model: Path) -> bool:
    test = model / "test"
    return (test / "summary.json").is_file() and (test / "results.json").is_file()


def run_command(cmd: Sequence[str], cwd: Path, log: Path, env: Mapping[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(subprocess.list2cmdline([x]) for x in cmd) + "\n")
        stream.flush()
        proc = subprocess.Popen(
            list(cmd), cwd=str(cwd), env=dict(env), stdout=stream, stderr=subprocess.STDOUT
        )
        rc = proc.wait()
        stream.write(f"\n[exit {rc}] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        stream.flush()
    if rc != 0:
        raise RuntimeError(f"command failed with exit {rc}; see {log}")


def percentile(values: Sequence[float], q: float) -> float:
    # Match NumPy's linear percentile without making NumPy a worker-only
    # dependency. NumPy is available in the official environment, but this
    # implementation keeps post-processing deterministic and lightweight.
    if not values:
        return float("nan")
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values) if values else float("nan")


def summarize_mode(
    records: Sequence[Mapping[str, Any]], mode: str, dataset: str, scene: str
) -> Dict[str, Any]:
    ae_key = f"{mode}_AE"
    te_key = f"{mode}_TE"
    aes = [float(r[ae_key]) for r in records]
    tes = [float(r[te_key]) for r in records]
    def recall(te: float, ae: float) -> float:
        return sum(1 for x, y in zip(tes, aes) if x <= te and y <= ae) / len(tes)
    return {
        "schema": "ulfloc_reproduction_mode_summary_v1",
        "dataset": dataset,
        "scene": scene,
        "mode": mode,
        "num_queries": len(records),
        "median_te_cm": percentile(tes, 50),
        "mean_te_cm": mean(tes),
        "p90_te_cm": percentile(tes, 90),
        "median_ae_deg": percentile(aes, 50),
        "mean_ae_deg": mean(aes),
        "p90_ae_deg": percentile(aes, 90),
        "recall_50cm_5deg": recall(50.0, 5.0),
        "recall_10cm_5deg": recall(10.0, 5.0),
        "recall_5cm_5deg": recall(5.0, 5.0),
        "recall_2cm_2deg": recall(2.0, 2.0),
        "recall_1cm_1deg": recall(1.0, 1.0),
        "catastrophic_100cm_count": sum(1 for x in tes if x >= 100.0),
    }


def postprocess(scene_root: Path, item: Mapping[str, str], repo: Path, gpu: str) -> Dict[str, Any]:
    model = model_dir(scene_root)
    official = read_json(model / "test" / "summary.json")
    raw = read_json(model / "test" / "results.json")
    sparse = summarize_mode(raw, "sparse", item["dataset"], item["scene"])
    dense = summarize_mode(raw, "dense", item["dataset"], item["scene"])
    sparse["official_summary"] = official.get("sparse", {})
    dense["official_summary"] = official.get("dense", {})
    sparse["model_path"] = str(model)
    dense["model_path"] = str(model)
    metrics = scene_root / "metrics"
    atomic_json(metrics / "sparse_summary.json", sparse)
    atomic_json(metrics / "dense_summary.json", dense)
    atomic_json(
        metrics / "combined_official_summary.json",
        {
            "schema": "ulfloc_reproduction_combined_summary_v1",
            "dataset": item["dataset"],
            "scene": item["scene"],
            "repo": str(repo),
            "gpu": str(gpu),
            "official": official,
            "sparse": sparse,
            "dense": dense,
        },
    )
    return {"sparse": sparse, "dense": dense}


def environment(repo: Path, gpu: str, python_bin: Path, experiment_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(repo) + os.pathsep + env.get("PYTHONPATH", ""),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "4"),
        }
    )
    # gsplat is JIT-compiled on the first training iteration.  Its default
    # torch-extension cache is process-shared, so launching several GPUs at
    # once can make concurrent ninja builds delete/replace one another's
    # objects.  Give each GPU in this matrix its own persistent cache and
    # bound compile parallelism; later scenes on the same GPU reuse it.
    cache_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment_root.name)
    env["TORCH_EXTENSIONS_DIR"] = f"/tmp/ulfloc_torch_extensions_{cache_tag}_gpu{gpu}"
    env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
    env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "8.6")
    # The conda cross-compiler shipped in this environment has an incomplete
    # glibc sysroot (and fails on crypt.h / bits headers).  Use the host GCC
    # pair for nvcc's host compilation; the Torch ABI flag is still supplied
    # by torch's extension builder, and conda's runtime libraries remain in
    # LD_LIBRARY_PATH below.
    env["CC"] = "/usr/bin/gcc"
    env["CXX"] = "/usr/bin/g++"
    env.pop("C_INCLUDE_PATH", None)
    env.pop("CPLUS_INCLUDE_PATH", None)
    # Keep the conda extension toolchain visible; this prevents the common
    # worker-only failure where ninja is present in the shell but not PATH.
    bin_dir = str(python_bin.parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    conda_lib = str(python_bin.parent.parent / "lib")
    env["LD_LIBRARY_PATH"] = conda_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def run_one(args: argparse.Namespace, item: Mapping[str, str]) -> int:
    root = Path(args.root).resolve()
    repo = Path(args.repo).resolve()
    python_bin = Path(args.python).resolve()
    scene_id = f"{item['dataset']}/{item['slug']}"
    scene_root = root / "scenes" / item["dataset"] / item["slug"]
    log = scene_root / "run.log"
    scene_root.mkdir(parents=True, exist_ok=True)
    stage = make_staged_scene(
        item,
        root / "staging",
        Path(args.data7).resolve(),
        Path(args.data12).resolve(),
        Path(args.cambridge).resolve(),
    )
    cfg = config_for(item, repo)
    model = model_dir(scene_root)
    env = environment(repo, str(args.gpu), python_bin, root)
    common = [str(python_bin)]

    train_cmd = common + [str(repo / "train.py"), "-s", str(stage), "-m", str(model)]
    train_cmd += ["--iterations", "30000", "--data_device", "cpu", "-f", "sp", "-g", "3dgs"]
    train_cmd += ["--sample_kpts", "--images", "images", "--cfg", str(cfg)]
    if item["dataset"] == "cambridge":
        train_cmd += [
            "-r", "1",
            "--densify_grad_threshold", "0.0004",
            "--position_lr_init", "0.000016",
            "--scaling_lr", "0.001",
        ]
    if not training_complete(model):
        run_command(train_cmd, repo, log, env)
    if not training_complete(model):
        raise RuntimeError(f"training exited but completion artifacts are missing: {model}")

    eval_cmd = common + [str(repo / "ulfloc.py"), "-s", str(stage), "-m", str(model)]
    eval_cmd += ["--data_device", "cpu", "--images", "images", "--cfg", str(cfg), "--longest_edge", "640"]
    if not evaluation_complete(model):
        run_command(eval_cmd, repo, log, env)
    if not evaluation_complete(model):
        raise RuntimeError(f"evaluation exited but summary/results are missing: {model / 'test'}")

    metrics = postprocess(scene_root, item, repo, str(args.gpu))
    marker = {
        "schema": "ulfloc_reproduction_scene_complete_v1",
        "scene_id": scene_id,
        "dataset": item["dataset"],
        "scene": item["scene"],
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "python": str(python_bin),
        "gpu": str(args.gpu),
        "stage": str(stage),
        "model": str(model),
        "sparse_summary": metrics["sparse"],
        "dense_summary": metrics["dense"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(scene_root / "scene_complete.json", marker)
    return 0


def initial_state(args: argparse.Namespace) -> Dict[str, Any]:
    repo = Path(args.repo).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return {
        "schema": "ulfloc_reproduction_matrix_state_v1",
        "repo": str(repo),
        "repo_commit": commit,
        "python": str(Path(args.python).resolve()),
        "root": str(Path(args.root).resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "datasets": {"7scenes": str(Path(args.data7).resolve()), "12scenes": str(Path(args.data12).resolve()), "cambridge": str(Path(args.cambridge).resolve())},
        "gpus": [str(x) for x in args.gpus],
        "scenes": {
            f"{x['dataset']}/{x['slug']}": {"dataset": x["dataset"], "scene": x["scene"], "status": "pending", "pid": None, "gpu": None, "attempts": 0, "last_error": None}
            for x in SCENES
        },
    }


def process_alive(pid: Any) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def coordinator(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep a discoverable coordinator PID for unattended monitoring.  It is
    # refreshed on every restart; scene state remains the authoritative
    # resumable record.
    atomic_text(root / "coordinator.pid", f"{os.getpid()}\n")
    state_path = root / "matrix_state.json"
    if state_path.exists() and not args.reset:
        state = read_json(state_path)
        if state.get("repo_commit") != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip():
            raise RuntimeError("existing matrix state belongs to a different ULF-Loc commit; use a new --root")
    else:
        state = initial_state(args)
        atomic_json(state_path, state)

    # A coordinator restart adopts live workers recorded in the state.  It
    # cannot waitpid an orphaned child, but it can wait for its atomic marker.
    running: Dict[str, subprocess.Popen] = {}
    for scene_id, rec in state["scenes"].items():
        if rec.get("status") == "running" and process_alive(rec.get("pid")):
            rec["status"] = "adopted_running"
    atomic_json(state_path, state)

    while True:
        # Reconcile adopted workers by their completion marker or dead PID.
        for scene_id, rec in state["scenes"].items():
            if rec.get("status") in {"running", "adopted_running"} and scene_id not in running:
                item = SCENE_BY_ID[scene_id]
                marker = root / "scenes" / item["dataset"] / item["slug"] / "scene_complete.json"
                if marker.exists():
                    rec.update({"status": "done", "pid": None})
                elif not process_alive(rec.get("pid")):
                    rec.update({"status": "failed", "pid": None, "last_error": "worker disappeared; rerun resumes this scene"})

        finished = []
        for scene_id, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            finished.append(scene_id)
            rec = state["scenes"][scene_id]
            rec["pid"] = None
            if rc == 0:
                rec["status"] = "done"
                rec["last_error"] = None
            else:
                rec["status"] = "failed_final" if args.no_retry or int(rec.get("attempts", 0)) >= int(args.max_retries) else "failed"
                rec["last_error"] = f"worker exit {rc}"
        for scene_id in finished:
            del running[scene_id]

        if not args.no_retry:
            for rec in state["scenes"].values():
                if rec.get("status") == "failed":
                    rec["status"] = "pending"

        # Include adopted workers when computing occupied devices.  This is
        # what prevents a coordinator restart from launching a second scene
        # on a GPU whose original child is still alive.
        occupied_gpus = {
            str(rec.get("gpu"))
            for rec in state["scenes"].values()
            if rec.get("status") in {"running", "adopted_running"} and process_alive(rec.get("pid"))
        }
        free_gpus = [str(x) for x in args.gpus if str(x) not in occupied_gpus]
        for scene_id, rec in state["scenes"].items():
            if not free_gpus or len(running) >= len(args.gpus):
                break
            if rec.get("status") != "pending":
                continue
            gpu = free_gpus.pop(0)
            item = SCENE_BY_ID[scene_id]
            cmd = [str(Path(args.python).resolve()), str(Path(__file__).resolve()), "--worker", "--scene-id", scene_id,
                   "--root", str(root), "--repo", str(Path(args.repo).resolve()), "--python", str(Path(args.python).resolve()),
                   "--data7", str(Path(args.data7).resolve()), "--data12", str(Path(args.data12).resolve()),
                   "--cambridge", str(Path(args.cambridge).resolve()), "--gpu", gpu]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log = root / "scenes" / item["dataset"] / item["slug"] / "coordinator.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("a", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]), env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            running[scene_id] = proc
            rec.update({"status": "running", "pid": proc.pid, "gpu": gpu, "attempts": int(rec.get("attempts", 0)) + 1, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state["counts"] = {status: sum(1 for r in state["scenes"].values() if r.get("status") == status) for status in ("pending", "running", "adopted_running", "done", "failed", "failed_final")}
        atomic_json(state_path, state)
        if not running and all(r.get("status") == "done" for r in state["scenes"].values()):
            print(json.dumps(state["counts"], indent=2))
            return 0
        if not running and not any(r.get("status") in {"pending", "failed", "running", "adopted_running"} for r in state["scenes"].values()):
            print(json.dumps(state["counts"], indent=2), file=sys.stderr)
            return 1
        time.sleep(max(5, int(args.poll_seconds)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/pool/sqy/ulfloc_reproduction_matrix_20260819")
    parser.add_argument("--repo", default="/tmp/ULF-Loc-repro-b28d532")
    parser.add_argument("--python", default="/root/miniconda3/envs/ulfloc_repro/bin/python")
    parser.add_argument("--data7", default="/mnt/pool/sqy/datasets/7Scenes_pgt_full_reference_v5")
    parser.add_argument("--data12", default="/mnt/pool/sqy/datasets/12Scenes_pgt_full_reference_v5")
    parser.add_argument("--cambridge", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scene-id")
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if not args.scene_id or args.scene_id not in SCENE_BY_ID:
            raise SystemExit(f"unknown --scene-id {args.scene_id!r}")
        try:
            return run_one(args, SCENE_BY_ID[args.scene_id])
        except Exception as exc:
            root = Path(args.root).resolve()
            item = SCENE_BY_ID[args.scene_id]
            atomic_json(
                root / "scenes" / item["dataset"] / item["slug"] / "worker_failure.json",
                {"scene_id": args.scene_id, "error": repr(exc), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
            )
            print(f"WORKER FAILURE: {exc}", file=sys.stderr)
            return 1
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
