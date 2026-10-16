"""Strict endpoint-only reference coverage CPU negatives."""
import copy
import unittest
import tempfile
from pathlib import Path
import project_a_reference as p

def raw_fixture():
    result = []
    for i in range(1, 9):
        light = {k: None for k in p.LIGHT_FIELDS}
        light.update(iteration=i, pool="dense", rng_digest="a" * 64, index=i, name="cpu_fixture",
            points=3, active_sh_degree=0, loss=0.5,
            loss_components={k: {"dtype": "torch.float32", "bits": "00000000"} for k in p.LOSS_FIELDS},
            lr={k: 0. for k in p.GROUPS},
            stacks={"viewpoint_idx_stack": None, "dense_viewpoint_idx_stack": []},
            events={"evaluation": [], "save": [], "native_checkpoint": []})
        for key in ("depth_order_prior_loss", "anisotropy_loss", "large_splat_loss"):
            light["loss_components"][key] = None
        light["loss_components"]["coefficients"] = {k: {"dtype": "float", "hex": "0x0.0p+0"} for k in
            ("lambda_normal", "lambda_dist", "lambda_anisotropy", "lambda_dense_depth")}
        result += [light, {"iteration": i, "exact_digest": str(i) * 64}]
        light["cadence_at_selection"] = 999999 if i == 1 else 5
        light["cadence_after_step"] = 5
        light["phase"] = None if i < 5 else "early"
        if i == 5:
            light["pool"] = "chart"
            for key in ("depth_order_prior_loss", "anisotropy_loss"):
                light["loss_components"][key] = {"dtype": "torch.float32", "bits": "00000000"}
            coeff = light["loss_components"]["coefficients"]
            del coeff["lambda_dense_depth"]
            for key in ("lambda_prior_depth", "lambda_prior_depth_derivative", "lambda_prior_normal",
                        "lambda_prior_curvature", "regularization_factor", "lambda_depth_order"):
                coeff[key] = {"dtype": "float", "hex": "0x0.0p+0"}
    return result

class ProjectionContracts(unittest.TestCase):
    def test_projection_preserves_exact_eight_light_and_endpoint(self):
        raw = raw_fixture()
        before = copy.deepcopy(raw)
        value = p.project_records(raw)
        self.assertEqual(len(value), 9)
        self.assertEqual(raw, before)
        self.assertEqual([r["iteration"] for r in value[:-1]], list(range(1, 9)))
        self.assertEqual(value[-1], raw[-1])
        p.validate_projected_records({p.validate_record(r): r for r in value})
    def test_raw_missing_extra_duplicate_and_reorder_rejected(self):
        raw = raw_fixture()
        bad_values = [raw[:-1], raw + [raw[-1]], raw[:3] + raw[4:],
                      [raw[1], raw[0]] + raw[2:]]
        for bad in bad_values:
            with self.assertRaises(RuntimeError):
                p.project_records(bad)
    def test_missing_or_extra_scalar_lr_stack_rng_rejected(self):
        mutations = [
            lambda r: r["loss_components"].pop("dssim"),
            lambda r: r["loss_components"].update(unknown={}),
            lambda r: r["loss_components"].update(Ll1=None),
            lambda r: r["lr"].pop("xyz"),
            lambda r: r["stacks"].pop("dense_viewpoint_idx_stack"),
            lambda r: r.update(rng_digest=""),
            lambda r: r.update(unknown_field=True),
        ]
        for mutate in mutations:
            raw = raw_fixture()
            mutate(raw[0])
            with self.assertRaises(RuntimeError):
                p.project_records(raw)
    def test_chart_components_and_cadence_are_required(self):
        for mutate in (lambda r: r["loss_components"].update(anisotropy_loss=None),
                       lambda r: r["loss_components"]["coefficients"].pop("lambda_depth_order"),
                       lambda r: r.update(phase=None),
                       lambda r: r.update(cadence_after_step=999999)):
            raw = raw_fixture()
            mutate(raw[8])
            with self.assertRaises(RuntimeError):
                p.project_records(raw)

    def test_independent_receipt_must_bind_actual_raw_trace_and_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "raw", Path(directory) / "launch"]
            for path in paths:
                path.write_text("fixture")
            receipt = {"evidence_sha256": {str(path): p.sha(path) for path in paths}}
            p.validate_review_links(receipt, *paths)
            for path in paths:
                bad = copy.deepcopy(receipt)
                del bad["evidence_sha256"][str(path)]
                with self.assertRaises(RuntimeError):
                    p.validate_review_links(bad, *paths)
            paths[0].write_text("changed")
            with self.assertRaises(RuntimeError):
                p.validate_review_links(receipt, *paths)

    def test_projected_no_extra_full_or_missing_endpoint_allowed(self):
        records = {p.validate_record(r): r for r in p.project_records(raw_fixture())}
        for key in ((1, False), (8, False), (8, True)):
            bad = dict(records)
            del bad[key]
            with self.assertRaises(RuntimeError):
                p.validate_projected_records(bad)
        bad = dict(records)
        bad[(1, True)] = {"iteration": 1, "exact_digest": "1" * 64}
        with self.assertRaises(RuntimeError):
            p.validate_projected_records(bad)

if __name__ == "__main__":
    unittest.main(verbosity=2)
