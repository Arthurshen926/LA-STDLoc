"""Generate a reversible, insertion-only adapter for the pinned native trainer."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re

SOURCE = Path("/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915/baselines/STDLoc/train.py")
EXPECTED = "80d4115a878441d8acf5a1ea0212440b753697e8464442d8a5d0946e321b9a10"
PROTOCOL_SHA = "75bb2ff18d072c8893c12818678bd0dc82f77ca0e8c74c2cf5e912d53645487a"


def strip_hooks(text):
    return re.sub(r"(?m)^ *# STD_EXACT_BEGIN [^\n]+\n.*?^ *# STD_EXACT_END [^\n]+\n",
                  "", text, flags=re.S)


def instrument(original):
    text = original
    edits = []
    def insert(anchor, code, label, before=False):
        nonlocal text
        if text.count(anchor) != 1:
            raise ValueError("nonunique frozen hook: " + label)
        indent = code[:len(code) - len(code.lstrip(" "))]
        block = indent + "# STD_EXACT_BEGIN " + label + "\n" + code + indent + "# STD_EXACT_END " + label + "\n"
        text = text.replace(anchor, block + anchor if before else anchor + block)
        edits.append(label)
    insert("from train_detector import training_detector\n",
           "from std_exact import Exact\n", "import")
    insert("    print(opt)\n", "    exact = Exact()\n", "controller", before=True)
    insert('    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Feature Gaussian")\n',
           '    restored = exact.prepare(locals())\n'
           '    if restored is not None:\n'
           '        first_iter = restored["first_iter"]\n'
           '        viewpoint_stack = restored["viewpoint_stack"]\n'
           '        ema_loss_for_log = restored["ema_loss_for_log"]\n'
           '        background = restored["background"]\n'
           '        del restored\n', "restore", before=True)
    insert("    first_iter += 1\n", "    exact.restore_rng_last(locals())\n", "rng_last")
    insert("                gaussians.optimizer.zero_grad(set_to_none=True)\n",
           "            if exact.after_step(locals()):\n"
           "                progress_bar.close()\n"
           "                return\n", "boundary")
    insert("    if train_detector:\n",
           "    exact.before_detector(locals())\n", "natural_detector", before=True)
    insert("            landmark_k=landmark_k,\n        )\n",
           "        exact.detector_complete()\n", "detector_done")
    if strip_hooks(text) != original:
        raise ValueError("hook stripping did not restore byte-identical native source")
    if ast.dump(ast.parse(strip_hooks(text))) != ast.dump(ast.parse(original)):
        raise ValueError("native AST changed")
    ast.parse(text)
    return text, edits


def build(root):
    root = Path(root)
    protocol = root / "training_protocol_v4.json"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("protocol v4 drift")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("native trainer drift")
    original = SOURCE.read_text()
    generated, edits = instrument(original)
    target = root / "std_execution"
    target.mkdir(exist_ok=True)
    trainer = target / "std_train_generated.py"
    if trainer.exists():
        raise FileExistsError(trainer)
    trainer.write_text(generated)
    (target / "std_trainer.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), generated.splitlines(True),
        fromfile=str(SOURCE), tofile=str(trainer))))
    here = Path(__file__).resolve().parent
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(here.glob("std_*.py"))}
    files[str(trainer)] = hashlib.sha256(trainer.read_bytes()).hexdigest()
    test_file = here.parents[1] / "tests/test_std_exact_stages.py"
    files[str(test_file)] = hashlib.sha256(test_file.read_bytes()).hexdigest()
    manifest = {"schema": "std_exact_build_v1", "complete": True,
                "protocol_sha256": PROTOCOL_SHA, "native_source": str(SOURCE),
                "native_source_sha256": EXPECTED, "generated_trainer": str(trainer),
                "file_sha256": files, "insertions": edits,
                "native_bytes_after_strip_identical": True,
                "native_ast_after_strip_identical": True, "gpu_executed": False}
    (target / "build.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    build(parser.parse_args().root)
