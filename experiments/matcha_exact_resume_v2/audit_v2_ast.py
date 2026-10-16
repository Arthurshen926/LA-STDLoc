"""Bounded source-only V1/V2/benchmark AST audit; never imports torch."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "matcha_exact_resume"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def functions(path):
    result = {}
    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            name = prefix
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if not isinstance(child, ast.ClassDef):
                    full = ast.dump(child, include_attributes=False)
                    body = copy.deepcopy(child)
                    body.decorator_list = []
                    result[name] = {"ast_sha256": hashlib.sha256(full.encode()).hexdigest(),
                        "without_decorators_sha256": hashlib.sha256(ast.dump(body, include_attributes=False).encode()).hexdigest(),
                        "line": child.lineno}
                name += "."
            visit(child, name)
    visit(ast.parse(path.read_text()))
    return result
def compare(old, new):
    a, b = functions(old), functions(new)
    changed, unchanged = [], []
    for name in sorted(set(a) | set(b)):
        if a.get(name, {}).get("ast_sha256") == b.get(name, {}).get("ast_sha256"):
            unchanged.append(name)
        else:
            changed.append({"function": name, "old": a.get(name), "new": b.get(name),
                "decorator_only": name in a and name in b and
                a[name]["without_decorators_sha256"] == b[name]["without_decorators_sha256"]})
    return {"unchanged": unchanged, "changed": changed}
def main():
    baseline = json.loads((ROOT / "V1_BASELINE_HASHES.json").read_text())
    assert all(sha(V1 / name) == digest for name, digest in baseline.items()), "V1 changed"
    reports, diff = {}, []
    for name in baseline:
        reports[name] = compare(V1 / name, ROOT / name)
        diff.extend(difflib.unified_diff((V1/name).read_text().splitlines(True),
            (ROOT/name).read_text().splitlines(True), fromfile="v1/"+name, tofile="v2/"+name))
    for name in ("train_with_charts_exact.py", "train_with_charts_observer.py", "build_trainer.py",
                 "build_observer.py", "continuation_gate.py", "test_launcher_cpu.py"):
        assert sha(ROOT/name) == baseline[name], "native trainer/resume gate changed: " + name
    old, new = functions(V1 / "exact_resume.py"), functions(ROOT / "exact_resume.py")
    for name in ("snapshot_static", "capture_model", "atomic_save", "read_atomic"):
        assert old[name]["without_decorators_sha256"] == new[name]["without_decorators_sha256"], name
    measured = ROOT / "benchmark_measured_exact_resume.py"
    bench = functions(measured)
    benchmark_equal = ["pack", "equal", "semantic_digest", "snapshot_static", "capture_model",
        "atomic_save", "read_atomic", "diagnostic_digest", "diagnostic_equal",
        "Controller.prepare", "Controller.verify_restored_boundary"]
    for name in benchmark_equal:
        assert bench[name]["ast_sha256"] == new[name]["ast_sha256"], "benchmark/final differs: " + name
    result = {"schema": "matcha_v2_function_ast_audit_v1", "verdict": "PASS",
        "v1_preserved": True, "native_trainers_builders_unchanged": True,
        "all_v1_function_differences": reports,
        "benchmark_measured_sha256": sha(measured),
        "benchmark_final_exact_functions": benchmark_equal,
        "benchmark_final_all_differences": compare(measured, ROOT/"exact_resume.py"),
        "new_modules": sorted(p.name for p in ROOT.glob("*.py") if p.name not in baseline),
        "source_sha256": {p.name: sha(p) for p in ROOT.glob("*.py")}}
    (ROOT/"function_ast_diff.json").write_text(json.dumps(result, sort_keys=True, indent=2))
    (ROOT/"v1_to_v2.diff").write_text("".join(diff))
    print(json.dumps({"audit_sha256": sha(ROOT/"function_ast_diff.json"),
                      "diff_sha256": sha(ROOT/"v1_to_v2.diff"),
                      "verdict": result["verdict"], "source_sha256": result["source_sha256"]}))
if __name__ == "__main__": main()
