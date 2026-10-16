"""Generate the isolated trainer; no imports of torch and no training."""
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
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED, "native source drift"
    text = original
    changes = []
    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, "nonunique frozen insertion: " + old[:80]
        text = text.replace(old, new)
        changes.append(old)
    replace("import json\n", 'import json\n'
            'sys.path.insert(0, "/root/MAtCha")\n'
            'sys.path.insert(0, "/root/MAtCha/2d-gaussian-splatting")\n'
            'from exact_resume import Controller, add_arguments\n')
    replace("    first_iter = 0\n", "    exact = Controller(args, locals())\n    first_iter = 0\n")
    replace('    if use_mip_filter:\n        print("[INFO] Using mip filter during training.")',
            '    if use_mip_filter and not exact.has_resume:\n        print("[INFO] Using mip filter during training.")')
    replace('    if use_dense_supervision and dense_regul != "none":\n',
            '    if use_dense_supervision and dense_regul != "none" and not exact.has_resume:\n')
    restore_names = (
        "background charts_scale_factor charts_prior_depths charts_depths charts_confs "
        "charts_normals charts_curvs chart_prior_weights chart_late_prior_weights "
        "chart_regularization_indices chart_late_regularization_indices dense_depth_priors "
        "use_dense_supervision use_depth_order_regularization viewpoint_idx_stack "
        "dense_viewpoint_idx_stack chart_sampling_phase use_chart_view_every_n_iter "
        "ema_loss_for_log ema_dist_for_log ema_normal_for_log ema_prior_depth_for_log "
        "ema_prior_normal_for_log ema_prior_curvature_for_log ema_prior_anisotropy_for_log "
        "first_iter").split()
    restore = '    restored = exact.prepare(locals())\n    if restored is not None:\n'
    restore += "".join('        %s = restored["%s"]\n' % (k, k) for k in restore_names)
    restore += '        del restored\n'
    replace('    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")\n',
            restore + '    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")\n')
    replace('    first_iter += 1\n', '    first_iter += 1\n    exact.restore_rng_last()\n    exact.verify_restored_boundary(locals())\n')
    replace('        render_pkg = render(viewpoint_cam, gaussians, pipe, background)\n',
            '        exact.observe_view(locals())\n        render_pkg = render(viewpoint_cam, gaussians, pipe, background)\n')
    anchor = '                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")\n'
    replace(anchor, anchor +
            '            if network_gui.conn is not None:\n'
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
    output = ROOT / "train_with_charts_exact.py"
    assert not output.exists(), "refusing generated source overwrite"
    output.write_text(text)
    diff = "".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                    fromfile=str(SOURCE), tofile=str(output), n=3))
    (ROOT / "trainer.diff").write_text(diff)
    (ROOT / "instrumentation.json").write_text(json.dumps({
        "schema": "matcha_exact_instrumentation_v1", "source": str(SOURCE),
        "source_sha256": EXPECTED, "generated": str(output),
        "generated_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "native_boundary": "after original lines 1943-1958; before original 1960 GUI block",
        "native_terminal": "30000 skips optimizer.step; terminal cannot be resumed",
        "replacement_count": len(changes), "gpu_executed": False,
    }, indent=2))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    print(json.dumps({"generated": str(output), "insertions": len(changes),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__ == "__main__":
    main()
