"""Generate observation-only native reference without any restore/skip branches."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
ROOT = Path(__file__).resolve().parent
SOURCE = Path("/root/MAtCha/2d-gaussian-splatting/train_with_charts.py")
EXPECTED = "5c5c7d924675dc3ad743208b3e2ca35d3d064f30892338845ff2c08a30b3a630"

def main():
    original = SOURCE.read_text()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    text = original
    def replace(a, b):
        nonlocal text
        assert text.count(a) == 1
        text = text.replace(a, b)
    replace("import json\n", 'import json\n'
        'sys.path.insert(0, "/root/MAtCha")\n'
        'sys.path.insert(0, "/root/MAtCha/2d-gaussian-splatting")\n'
        'from exact_resume import Controller, add_arguments\n')
    replace("    first_iter = 0\n", '    if args.exact_resume is not None:\n'
        '        raise RuntimeError("native observer never restores checkpoints")\n'
        '    exact = Controller(args, locals())\n    first_iter = 0\n')
    anchor = '    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")\n'
    replace(anchor, '    assert exact.prepare(locals()) is None\n' + anchor)
    replace('    first_iter += 1\n', '    first_iter += 1\n    exact.restore_rng_last()\n    exact.verify_restored_boundary(locals())\n')
    replace('        render_pkg = render(viewpoint_cam, gaussians, pipe, background)\n',
            '        exact.observe_view(locals())\n        render_pkg = render(viewpoint_cam, gaussians, pipe, background)\n')
    anchor = '                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")\n'
    replace(anchor, anchor + '            if network_gui.conn is not None:\n'
        '                raise RuntimeError("exact-resume: live GUI is unsupported")\n'
        '            if exact.boundary(locals()):\n'
        '                progress_bar.close()\n'
        '                break\n')
    replace('    args = parser.parse_args(sys.argv[1:])\n',
            '    add_arguments(parser)\n    args = parser.parse_args(sys.argv[1:])\n')
    replace('                network_gui.try_connect(dataset.render_items)\n',
            '                network_gui.try_connect(dataset.render_items)\n'
            '                if network_gui.conn is not None:\n'
            '                    raise RuntimeError("exact-resume: GUI connection refused before render")\n')
    replace('        iter_start.record()\n',
            '        exact.before_step(iteration)\n        iter_start.record()\n')
    replace('    return {"Ll1": Ll1, "loss": loss}\n',
            '    return {"Ll1": Ll1, "loss": loss, "dssim": dssim if lambda_dssim > 0.0 else None}\n')
    ast.parse(text)
    out = ROOT / "train_with_charts_observer.py"
    assert not out.exists()
    out.write_text(text)
    (ROOT / "observer.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), text.splitlines(True), fromfile=str(SOURCE), tofile=str(out))))
    print(json.dumps({"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "can_resume": False, "gpu_executed": False}))
if __name__ == "__main__":
    main()
