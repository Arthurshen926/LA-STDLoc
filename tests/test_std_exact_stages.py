"""CPU-only state, resume, schedule and admission tests for STDLoc adapter."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/native_baseline_repair"))

import numpy as np
import torch
import std_build as build
import std_exact as exact
import std_launch as launch


class FakeGaussian(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in exact.PARAMETERS:
            setattr(self, name, torch.nn.Parameter(torch.randn(3, 2)))
        self.active_sh_degree = 2
        self.max_sh_degree = 3
        self.spatial_lr_scale = 2.5
        self.percent_dense = 0.01
        self.max_radii2D = torch.randn(3)
        self.xyz_gradient_accum = torch.randn(3, 1)
        self.denom = torch.randn(3, 1)
        self.setup()

    def setup(self):
        self.optimizer = torch.optim.Adam(
            [{"params": [p], "lr": 0.01, "name": name}
             for name, p in self.named_parameters()], eps=1e-15)

    def capture(self):
        return (self.active_sh_degree, self._xyz, self._features_dc,
                self._features_rest, self._scaling, self._rotation, self._opacity,
                self.max_radii2D, self.xyz_gradient_accum, self.denom,
                self.optimizer.state_dict(), self.spatial_lr_scale, self._loc_feature)

    def restore(self, values, opt):
        (self.active_sh_degree, self._xyz, self._features_dc,
         self._features_rest, self._scaling, self._rotation, self._opacity,
         self.max_radii2D, self.xyz_gradient_accum, self.denom, optimizer,
         self.spatial_lr_scale, self._loc_feature) = values
        self.setup()
        self.optimizer.load_state_dict(optimizer)


class FakeScene:
    def __init__(self):
        self.cameras = []
        for i in range(5):
            fields = {name: float(i) for name in exact.CAMERA_FIELDS}
            fields.update(image_name="camera_" + str(i), uid=i, colmap_id=i,
                          R=np.eye(3), T=np.zeros(3),
                          world_view_transform=torch.eye(4),
                          projection_matrix=torch.eye(4),
                          full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
            self.cameras.append(SimpleNamespace(**fields))
        self.cameras_extent = 3.0
        self.resolution_scales = [1.0]
        self.feature_type = "sp"
        self.longest_edge = 640

    def getTrainCameras(self):
        return self.cameras

    def getTestCameras(self):
        return self.cameras[:1]


def namespace():
    scene = FakeScene()
    return {"gaussians": FakeGaussian(), "feature_extractor": torch.nn.Linear(2, 2).eval(),
            "scene": scene, "opt": SimpleNamespace(iterations=30000, position_lr_init=0.000016),
            "dataset": SimpleNamespace(model_path="synthetic", gaussian_type="3dgs"),
            "background": torch.zeros(3), "viewpoint_stack": None, "ema_loss_for_log": 0.0,
            "iteration": 0}


def synthetic_step(ns, iteration):
    if not ns["viewpoint_stack"]:
        ns["viewpoint_stack"] = ns["scene"].getTrainCameras().copy()
    cam = ns["viewpoint_stack"].pop(random.randint(0, len(ns["viewpoint_stack"]) - 1))
    g, f = ns["gaussians"], ns["feature_extractor"]
    x = torch.randn(3, 2) + np.random.random() + cam.uid
    loss = sum((p * f(x)).square().mean() for p in g.parameters())
    loss.backward()
    if iteration < 30000:
        g.optimizer.step()
        g.optimizer.zero_grad(set_to_none=True)
    ns["iteration"] = iteration
    ns.update(loss=loss, Ll1=loss, Ll1_feature=loss, viewpoint_cam=cam)
    ns["ema_loss_for_log"] = 0.4 * loss.item() + 0.6 * ns["ema_loss_for_log"]
    return cam.image_name, loss.item()


class ExactTests(unittest.TestCase):
    def setUp(self):
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)

    def test_cpu_only(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_rng_streams_roundtrip(self):
        state = exact.rng_state()
        a = (random.random(), np.random.rand(), torch.randn(3))
        exact.set_rng(state)
        b = (random.random(), np.random.rand(), torch.randn(3))
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))

    def test_full_resume_matches_uninterrupted_with_accumulated_feature_grad(self):
        ns = namespace()
        for i in range(1, 13):
            synthetic_step(ns, i)
        contract = exact.scene_contract(ns["scene"])
        identity = {"case": "CPU synthetic"}
        state = exact.capture_state(ns, contract, identity)
        boundary_manifest = exact.tree_manifest(state)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "checkpoint"
            exact.publish_checkpoint(directory, state)
            continuous = [synthetic_step(ns, i) for i in range(13, 18)]
            expected = exact.tree_manifest(exact.capture_state(ns, contract, identity))
            restored, record = exact.load_checkpoint(directory)
            self.assertEqual(exact.tree_manifest(restored), boundary_manifest)
            other = namespace()  # deliberately consumes RNG and resets feature gradients
            result = exact.restore_state(restored, other, exact.scene_contract(other["scene"]), identity)
            other.update(result)
            other["iteration"] = result["first_iter"]
            exact.set_rng(restored["rng"])
            self.assertEqual(exact.tree_manifest(exact.capture_state(other, contract, identity)),
                             boundary_manifest)
            actual = [synthetic_step(other, i) for i in range(13, 18)]
            self.assertEqual(actual, continuous)
            self.assertEqual(exact.tree_manifest(exact.capture_state(other, contract, identity)), expected)
            self.assertTrue(all(p.grad is not None for p in other["feature_extractor"].parameters()))

    def test_terminal_gradients_are_preserved(self):
        ns = namespace()
        before = [p.detach().clone() for p in ns["gaussians"].parameters()]
        synthetic_step(ns, 30000)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, ns["gaussians"].parameters())))
        self.assertTrue(all(p.grad is not None for p in ns["gaussians"].parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            exact.publish_checkpoint(Path(tmp) / "cp",
                                     exact.capture_state(ns, exact.scene_contract(ns["scene"]), {}))
            state, _ = exact.load_checkpoint(Path(tmp) / "cp")
            other = namespace()
            exact.restore_state(state, other, exact.scene_contract(other["scene"]), {})
            for a, b in zip(ns["gaussians"].parameters(), other["gaussians"].parameters()):
                self.assertTrue(torch.equal(a.grad, b.grad))

    def test_blob_tampering_fails_before_load(self):
        ns = namespace()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp"
            exact.publish_checkpoint(cp, exact.capture_state(ns, exact.scene_contract(ns["scene"]), {}))
            with (cp / "state.pt").open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaisesRegex(ValueError, "blob drift"):
                exact.load_checkpoint(cp)

    def test_incomplete_and_duplicate_publication_fail(self):
        ns = namespace()
        state = exact.capture_state(ns, exact.scene_contract(ns["scene"]), {})
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp"
            cp.mkdir()
            with self.assertRaises(FileNotFoundError):
                exact.load_checkpoint(cp)
            with self.assertRaises(FileExistsError):
                exact.publish_checkpoint(cp, state)
            exact.atomic_json(Path(tmp) / "marker", {"first": True})
            with self.assertRaises(FileExistsError):
                exact.atomic_json(Path(tmp) / "marker", {"first": False})
            self.assertTrue(json.loads((Path(tmp) / "marker").read_text())["first"])

    def test_contract_and_identity_drift_fail(self):
        ns = namespace()
        contract = exact.scene_contract(ns["scene"])
        state = exact.capture_state(ns, contract, {"id": 1})
        with self.assertRaisesRegex(ValueError, "identity"):
            exact.restore_state(state, ns, contract, {"id": 2})
        changed = copy.deepcopy(contract)
        changed["train"].reverse()
        with self.assertRaisesRegex(ValueError, "scene"):
            exact.restore_state(state, ns, changed, {"id": 1})
        ns["opt"].iterations = 20000
        with self.assertRaisesRegex(ValueError, "options"):
            exact.restore_state(state, ns, contract, {"id": 1})

    def test_camera_stack_order_and_empty_differ(self):
        scene = FakeScene()
        stack = [scene.cameras[3], scene.cameras[1]]
        self.assertEqual(exact.stack_indices(stack, scene), [3, 1])
        self.assertEqual(exact.stack_from_indices([3, 1], scene), stack)
        self.assertIsNone(exact.stack_indices(None, scene))
        self.assertEqual(exact.stack_indices([], scene), [])
        with self.assertRaises(ValueError):
            exact.stack_from_indices([1, 1], scene)
        with self.assertRaises(ValueError):
            exact.stack_from_indices([99], scene)

    def test_tensor_manifest_detects_bytes_dtype_and_grad(self):
        tensor = torch.tensor([0.0, -0.0])
        self.assertNotEqual(exact.digest(tensor), exact.digest(torch.zeros(2)))
        self.assertNotEqual(exact.digest(tensor), exact.digest(tensor.double()))
        self.assertNotEqual(exact.digest(tensor), exact.digest(tensor.requires_grad_()))

    def test_insertion_only_frozen_native_ast(self):
        source = build.SOURCE.read_text()
        self.assertEqual(exact.sha(build.SOURCE), build.EXPECTED)
        generated, labels = build.instrument(source)
        self.assertEqual(build.strip_hooks(generated), source)
        self.assertEqual(len(labels), 7)
        self.assertIn("if iteration < opt.iterations:", generated)
        self.assertIn("train_iteration=30000,", generated)
        self.assertIn("seed_everything(2025)", generated)
        with self.assertRaises(ValueError):
            build.instrument(source.replace("    first_iter += 1\n", ""))

    def test_lease_nofile_and_reviews_fail_closed(self):
        lease = {"approved": True, "scope": "std_exact_native_stage",
                 "protocol_sha256": build.PROTOCOL_SHA, "build_sha256": "abc",
                 "mapping_semantics": "off", "stage": "reference",
                 "approval_reference": "approval", "owner_handover": "handover",
                 "gpu_uuid": "GPU-test", "valid_from_unix": 100, "valid_until_unix": 4000}
        self.assertEqual(launch.check_lease(lease, "abc", "off", "reference", now=200), "GPU-test")
        for key, value in (("stage", "gaussian_1"), ("build_sha256", "wrong"),
                           ("valid_until_unix", 300), ("approved", False)):
            changed = dict(lease, **{key: value})
            with self.assertRaises(ValueError):
                launch.check_lease(changed, "abc", "off", "reference", now=200)
        self.assertEqual(launch.nofile_pair(1024, 1048576), (4096, 1048576))
        with self.assertRaises(ValueError):
            launch.nofile_pair(1024, 2048)
        with self.assertRaises(ValueError):
            launch.check_review({}, "abc", "off", "gaussian_1")

    def test_controller_boundary_restores_at_next_native_step(self):
        ns = namespace()
        ns.update(checkpoint=None, train_detector=True)
        for i in range(1, 4):
            synthetic_step(ns, i)
        ns["iteration"] = 10000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"identity": {"case": 1}, "start": 0, "stop": 10000,
                      "run_dir": str(root), "stage": "gaussian_1",
                      "checkpoint_out": str(root / "cp")}
            path = root / "config"
            path.write_text(json.dumps(config))
            previous = os.environ.get("STD_EXACT_CONFIG")
            os.environ["STD_EXACT_CONFIG"] = str(path)
            try:
                controller = exact.Exact()
                self.assertIsNone(controller.prepare(ns))
                controller.last_step = 9999
                self.assertTrue(controller.after_step(ns))
                self.assertTrue(json.loads((root / "gaussian_complete.json").read_text())["complete"])
                config.update(start=10000, stop=20000, stage="gaussian_2", resume=str(root / "cp"))
                path.write_text(json.dumps(config))
                other = namespace()
                other.update(checkpoint=None, train_detector=True)
                controller = exact.Exact()
                other.update(controller.prepare(other))
                other["first_iter"] += 1  # precisely the original trainer ordering
                controller.restore_rng_last(other)
                self.assertEqual(other["first_iter"], 10001)
                self.assertIsNone(controller.state)
                self.assertTrue(json.loads((root / "restore_verified.json").read_text())["complete"])
                with self.assertRaises(ValueError):
                    controller.before_detector(other)
            finally:
                if previous is None:
                    os.environ.pop("STD_EXACT_CONFIG", None)
                else:
                    os.environ["STD_EXACT_CONFIG"] = previous

    def test_final_receipt_requires_all_four_stages_and_native_artifacts(self):
        import std_entry
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = {"probe_only": False, "stage": "detector", "run_dir": str(base / "detector"),
                      "identity": {"model_path": str(base / "model"), "protocol_sha256": "p", "build_sha256": "b"},
                      "final_receipt": str(base / "receipts/final.json")}
            for stage in ("gaussian_1", "gaussian_2", "gaussian_3", "detector"):
                (base / stage).mkdir()
                start, stop = launch.STAGES[stage]
                record = {"status": "complete", "probe_only": False, "exact_stage": stage,
                          "gaussian_start": start, "gaussian_stop": stop,
                          "protocol_sha256": "p", "build_sha256": "b",
                          "wall_seconds": 10, "peak_rss_kib": 1, "peak_torch_gpu_bytes": 2,
                          "build_stages": []}
                (base / stage / "stage_complete.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "missing native"):
                std_entry.finalize_training(config)
            for relative in ("point_cloud/iteration_30000/point_cloud.ply",
                             "detector/sampled_idx.pkl", "detector/30000_detector.pth"):
                path = base / "model" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic")
            std_entry.finalize_training(config)
            receipt = json.loads(Path(config["final_receipt"]).read_text())
            self.assertEqual(receipt["wall_seconds"], 40)
            self.assertEqual(receipt["native_total_iterations"], 30000)
            self.assertEqual(receipt["native_optimizer_steps"], 29999)
            with self.assertRaises(ValueError):
                std_entry.finalize_training(dict(config, probe_only=True))

    def test_numeric_report_uses_only_observed_native_aa_envelope(self):
        import std_compare
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            identity = {"protocol_sha256": "p", "build_sha256": "b", "mapping_semantics": "off"}
            for j, name in enumerate(("reference", "probe_a", "probe_b")):
                directory = base / name
                directory.mkdir()
                (directory / "config.json").write_text(json.dumps({"identity": identity, "probe_only": True}))
                (directory / "stage_complete.json").write_text(json.dumps({"status": "complete", "probe_only": True}))
                trace = [{"iteration": i, "camera": "x", "rng_digest": "r", "optimizer_lr": [0.001],
                          "loss": 1.0 + 0.01 * j, "l1": 0.25 + 0.01 * j,
                          "feature_l1": 0.75 + 0.01 * j} for i in range(10001, 10033)]
                (directory / "trace.json").write_text(json.dumps(trace))
                if j:
                    (directory / "restore_verified.json").write_text(json.dumps(
                        {"complete": True, "iteration": 10000, "state_digest": "s"}))
            cp = base / "reference/checkpoint_10000"
            cp.mkdir()
            (cp / "complete.json").write_text(json.dumps(
                {"complete": True, "iteration": 10000, "state_digest": "s"}))
            report = std_compare.compare(base)
            self.assertFalse(report["within_observed_native_repeatability"])
            self.assertAlmostEqual(report["observed_native_aa_envelopes_abs"]["loss"], 0.01)
            self.assertNotIn("verdict", report)
            self.assertTrue(report["independent_review_still_required"])

    def test_masks_are_bitwise_checked_after_reconstruction(self):
        ns = namespace()
        ns["masks"] = {c.image_name: tuple(torch.ones(2, 3, dtype=torch.bool) for _ in range(3))
                       for c in ns["scene"].getTrainCameras()}
        state = exact.capture_state(ns, exact.scene_contract(ns["scene"]), {})
        ns["masks"]["camera_2"][1][0, 0] = False
        with self.assertRaisesRegex(ValueError, "masks"):
            exact.restore_state(state, ns, exact.scene_contract(ns["scene"]), {})

    def test_cpu_roundtrip_preserves_serialized_cuda_device_metadata_without_cuda(self):
        import io
        from unittest.mock import patch
        original = list(torch.serialization._package_registry)
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
        fake_cuda = (0, lambda storage: "cuda:0", lambda storage, location: None)
        with patch.object(torch.serialization, "_package_registry", [fake_cuda] + original):
            stream = io.BytesIO()
            torch.save(tensor, stream)
        stream.seek(0)
        locations = {}
        def on_cpu(storage, location):
            locations[storage._cdata] = location
            return storage
        restored = torch.load(stream, weights_only=False, map_location=on_cpu)
        self.assertEqual(exact.tree_manifest(restored, locations),
                         exact.tree_manifest(tensor, {tensor.untyped_storage()._cdata: "cuda:0"}))
        self.assertEqual(str(restored.device), "cpu")
        self.assertFalse(torch.cuda.is_initialized())

    def test_nonfinite_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            exact.require_finite_tensors({"optimizer": [torch.tensor(float("nan"))]})

    def test_restore_rejects_parameter_gradient_and_adam_alias_mutation(self):
        ns = namespace()
        for i in range(1, 4):
            synthetic_step(ns, i)
        synthetic_step(ns, 30000)  # terminal gradients remain live
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "cp"
            exact.publish_checkpoint(checkpoint, exact.capture_state(
                ns, exact.scene_contract(ns["scene"]), {"case": "alias"}))
            previous = os.environ.get("STD_EXACT_CONFIG")
            try:
                for target in ("parameter", "gradient", "adam"):
                    run = root / target
                    run.mkdir()
                    config = {"identity": {"case": "alias"}, "start": 30000, "stop": 30000,
                              "run_dir": str(run), "stage": "detector", "resume": str(checkpoint)}
                    path = run / "config.json"
                    path.write_text(json.dumps(config))
                    os.environ["STD_EXACT_CONFIG"] = str(path)
                    other = namespace()
                    other.update(checkpoint=None, train_detector=True)
                    controller = exact.Exact()
                    other.update(controller.prepare(other))
                    other["first_iter"] += 1
                    parameter = other["gaussians"]._xyz
                    with torch.no_grad():
                        if target == "parameter":
                            parameter[0, 0] += 1
                        elif target == "gradient":
                            parameter.grad[0, 0] += 1
                        else:
                            other["gaussians"].optimizer.state[parameter]["exp_avg"][0, 0] += 1
                    with self.assertRaisesRegex(ValueError, "not byte-exact"):
                        controller.restore_rng_last(other)
                    self.assertFalse((run / "restore_verified.json").exists())
            finally:
                if previous is None:
                    os.environ.pop("STD_EXACT_CONFIG", None)
                else:
                    os.environ["STD_EXACT_CONFIG"] = previous

    def test_exact_stage_schedule(self):
        self.assertEqual([launch.STAGES[x] for x in ("gaussian_1", "gaussian_2", "gaussian_3", "detector")],
                         [(0, 10000), (10000, 20000), (20000, 30000), (30000, 30000)])


if __name__ == "__main__":
    unittest.main()
