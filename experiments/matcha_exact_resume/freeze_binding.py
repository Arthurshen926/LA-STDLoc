"""Freeze bindings only AFTER independent corrected-input review.

This script hashes real training inputs and compiled extensions, so it is NOT a
micro CPU test and must wait for the parent's timing/resource approval.
It does not launch training or initialize CUDA.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PRODUCER = Path("/root/MAtCha")
NATIVE = PRODUCER / "2d-gaussian-splatting/train_with_charts.py"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def add_bound_file(files, path, digest):
    path = str(path)
    assert path not in files or files[path] == digest, "conflicting frozen SHA: " + path
    files[path] = digest

def parsed_science(argv):
    # Parse the unchanged native argparse block without importing renderer/torch.
    spec = importlib.util.spec_from_file_location("native_arguments", PRODUCER / "2d-gaussian-splatting/arguments/__init__.py")
    args_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(args_module)
    tree = ast.parse(NATIVE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If) and
                isinstance(n.test, ast.Compare) and ast.unparse(n.test) == "__name__ == '__main__'")
    body = []
    for n in main.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "print":
            break
        body.append(n)
    env = {k: getattr(args_module, k) for k in ("ModelParams", "OptimizationParams", "PipelineParams")}
    env.update(ArgumentParser=argparse.ArgumentParser, sys=type("Argv", (), {"argv": argv})())
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(NATIVE), "exec"), env)
    ignored = {"model_path", "quiet", "port"}
    return {k: v for k, v in vars(env["args"]).items() if k not in ignored}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-plan", required=True, type=Path)
    parser.add_argument("--input-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    assert not a.output.exists(), "refusing binding overwrite"
    plan = json.loads(a.training_plan.read_text())
    review = json.loads(a.input_review.read_text())
    assert review["schema"] == "matcha_resume_input_review_v1" and review["verdict"] == "PASS"
    assert review["training_plan_sha256"] == sha(a.training_plan)
    assert sha(NATIVE) == "5c5c7d924675dc3ad743208b3e2ca35d3d064f30892338845ff2c08a30b3a630"
    files = dict(plan["bound_source_hashes"])
    for path, expected in files.items():
        assert sha(path) == expected, "stale original plan binding: " + path
    for rel, digest in plan["input_artifact_hashes"].items():
        add_bound_file(files, str((a.training_plan.parent / rel).resolve()), digest)
    for path, info in plan["symlink_input_hashes"].items():
        add_bound_file(files, path, info["sha256"])
    for path in (a.training_plan, a.input_review, ROOT / "exact_resume.py",
                 ROOT / "train_with_charts_exact.py", ROOT / "build_trainer.py",
                 ROOT / "freeze_binding.py", ROOT / "build_observer.py",
                 ROOT / "train_with_charts_observer.py", ROOT / "continuation_gate.py",
                 ROOT / "native_n8_launcher.py"):
        add_bound_file(files, str(path.resolve()), sha(path))
    # Bind the original code tree and local compiled CUDA operators actually used.
    for folder in (PRODUCER / "2d-gaussian-splatting", PRODUCER / "matcha", PRODUCER / "Depth-Anything-V2/depth_anything_v2"):
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".so", ".yaml"):
                add_bound_file(files, str(p.resolve()), sha(p))
    site = Path("/root/miniconda3/envs/cybersim_agent/lib/python3.9/site-packages")
    for module in ("simple_knn", "diff_surfel_rasterization", "diff_gaussian_rasterization"):
        for p in (site / module).glob("*.so"):
            add_bound_file(files, str(p.resolve()), sha(p))
    for flag in ("--cambridge_mask_pickle",):
        p = Path(plan["argv"][plan["argv"].index(flag) + 1])
        add_bound_file(files, str(p.resolve()), sha(p))
    checkpoint_dir = Path(plan["argv"][plan["argv"].index("--depthanythingv2_checkpoint_dir") + 1])
    encoder = plan["argv"][plan["argv"].index("--depthanything_encoder") + 1]
    checkpoint = checkpoint_dir / ("depth_anything_v2_" + encoder + ".pth")
    assert checkpoint.is_file(), "verify the actual native Depth-Anything checkpoint path"
    add_bound_file(files, str(checkpoint.resolve()), sha(checkpoint))
    for path, expected in files.items():
        assert sha(path) == expected, "file changed: " + path
    output = {"schema": "matcha_exact_binding_v1", "files": files,
              "input_receipt_sha256": sha(a.input_review),
              "science_args": parsed_science(plan["argv"])}
    a.output.write_text(json.dumps(output, sort_keys=True, indent=2))
    print(json.dumps({"output": str(a.output), "files": len(files), "sha256": sha(a.output)}))

if __name__ == "__main__":
    main()
