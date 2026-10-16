"""Keep the frozen protocol wrapper intact; redirect only its native train path."""
import json
import os
from pathlib import Path
import runpy
import sys

from std_exact import atomic_json, sha


def finalize_training(config):
    """Publish the final requested receipt only after all four native stages."""
    if config["probe_only"] or config["stage"] != "detector":
        raise ValueError("diagnostics cannot publish a final model receipt")
    base = Path(config["run_dir"]).parent
    records = []
    references = []
    for stage, bounds in (("gaussian_1", (0, 10000)), ("gaussian_2", (10000, 20000)),
                          ("gaussian_3", (20000, 30000)), ("detector", (30000, 30000))):
        path = base / stage / "stage_complete.json"
        record = json.loads(path.read_text())
        if (record.get("status") != "complete" or record.get("probe_only")
                or record.get("exact_stage") != stage
                or (record["gaussian_start"], record["gaussian_stop"]) != bounds
                or record.get("protocol_sha256") != config["identity"]["protocol_sha256"]
                or record.get("build_sha256") != config["identity"]["build_sha256"]):
            raise ValueError("invalid completed stage: " + stage)
        records.append(record)
        references.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    model = Path(config["identity"]["model_path"])
    artifacts = []
    for relative in ("point_cloud/iteration_30000/point_cloud.ply",
                     "detector/sampled_idx.pkl", "detector/30000_detector.pth"):
        path = model / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("missing native terminal artifact: " + str(path))
        artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    value = dict(records[-1])
    value.update({"scope": "all_four_native_stages", "exact_stage": "all_complete",
                  "wall_seconds": sum(r["wall_seconds"] for r in records),
                  "peak_rss_kib": max(r["peak_rss_kib"] for r in records),
                  "peak_torch_gpu_bytes": max(r["peak_torch_gpu_bytes"] for r in records),
                  "build_stages": [s for r in records for s in r["build_stages"]],
                  "native_total_iterations": 30000, "native_optimizer_steps": 29999,
                  "native_terminal_skips_optimizer_step": True,
                  "detector_train_iterations": 30000, "landmark_num": 16384, "landmark_k": 32,
                  "native_output": str(model), "stage_receipts": references, "artifacts": artifacts,
                  "timing_protocol": "sum_of_runner_stage_native_wrapper_wall_times_includes_repeated_loader_and_checkpoint_overhead; not_isolated_benchmark",
                  "timing_excludes": "launcher_preflight_and_final_artifact_hashing",
                  "build_cost_scope": "joint_gaussian_and_feature_training_plus_detector",
                  "gaussian_start": 0, "gaussian_stop": 30000})
    destination = Path(config["final_receipt"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, value)


def main():
    config_path = Path(os.environ["STD_EXACT_CONFIG"])
    if sha(config_path) != os.environ["STD_EXACT_CONFIG_SHA256"]:
        raise ValueError("stage config drift")
    config = json.loads(config_path.read_text())
    for path, expected in config["code_sha256"].items():
        if sha(path) != expected:
            raise ValueError("code changed after launcher: " + path)
    native = config["native_trainer"]
    generated = config["generated_trainer"]
    original_run_path = runpy.run_path
    invoked = []
    def run_path(path, *args, **kwargs):
        if str(path) == native:
            if invoked or kwargs.get("run_name") != "__main__":
                raise ValueError("unexpected native train entry")
            invoked.append(path)
            return original_run_path(generated, *args, **kwargs)
        return original_run_path(path, *args, **kwargs)
    runpy.run_path = run_path
    command = config["wrapper_command"]
    sys.argv = command[1:]
    try:
        original_run_path(command[1], run_name="__main__")
    finally:
        runpy.run_path = original_run_path
    if len(invoked) != 1:
        raise ValueError("native trainer was not invoked exactly once")
    run_dir = Path(config["run_dir"])
    receipt = json.loads(Path(config["stage_receipt"]).read_text())
    marker = run_dir / ("detector_complete.json" if config["stage"] == "detector" else "gaussian_complete.json")
    if receipt.get("status") != "complete" or not json.loads(marker.read_text()).get("complete"):
        raise ValueError("native stage not complete")
    receipt.update({"exact_stage": config["stage"], "native_total_iterations": 30000,
                    "gaussian_start": config["start"], "gaussian_stop": config["stop"],
                    "probe_only": config.get("probe_only", False),
                    "protocol_sha256": config["identity"]["protocol_sha256"],
                    "build_sha256": config["identity"]["build_sha256"],
                    "scope": "single_stage_excludes_other_stages"})
    atomic_json(run_dir / "stage_complete.json", receipt)
    if config["stage"] == "detector":
        finalize_training(config)


if __name__ == "__main__":
    main()
