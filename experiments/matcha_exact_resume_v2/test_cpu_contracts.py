"""Micro CPU contracts only. No production arrays, inputs, renderer, or CUDA."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import ast
import __future__
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import exact_resume as er

SOURCE = Path("/root/MAtCha/2d-gaussian-splatting")
def native_function(path, name, globals_):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    node = copy.deepcopy(nodes[0])
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), globals_)
    return globals_[name]

def native_cpu_setup():
    tree = ast.parse((SOURCE / "scene/gaussian_model.py").read_text())
    node = copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "training_setup"))
    class DeviceOnly(ast.NodeTransformer):
        replacements = 0
        def visit_keyword(self, n):
            if n.arg == "device" and isinstance(n.value, ast.Constant) and n.value.value == "cuda":
                n.value.value = "cpu"
                self.replacements += 1
            return n
    transform = DeviceOnly()
    node = transform.visit(node)
    assert transform.replacements == 2
    env = {"torch": torch, "get_expon_lr_func": native_function(
        SOURCE / "utils/general_utils.py", "get_expon_lr_func", {"np": np})}
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "native_training_setup_CPU_device_only", "exec"), env)
    return env["training_setup"]

class Model:
    training_setup = native_cpu_setup()
    update_learning_rate = native_function(SOURCE / "scene/gaussian_model.py", "update_learning_rate", {})
    @property
    def get_xyz(self):
        return self._xyz
    def __init__(self, opt):
        shapes = {"xyz": (3, 3), "f_dc": (3, 1, 3), "f_rest": (3, 15, 3),
                  "opacity": (3, 1), "scaling": (3, 2), "rotation": (3, 4)}
        for name, attr in er.PARAMS.items():
            setattr(self, attr, torch.nn.Parameter(torch.randn(shapes[name])))
        self.spatial_lr_scale = 0.107
        self.active_sh_degree, self.max_sh_degree = 1, 3
        self.max_radii2D = torch.arange(3, dtype=torch.float32)
        self.use_mip_filter = True
        self.mip_filter = torch.tensor([[0.01], [0.02], [0.03]])
        self.training_setup(opt)

def options():
    return types.SimpleNamespace(iterations=30000, position_lr_init=0.00016,
        position_lr_final=0.0000016, position_lr_delay_mult=0.01, position_lr_max_steps=30000,
        feature_lr=0.0025, opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001,
        percent_dense=0.01, lambda_dssim=0.2, non_position_lr_decay_from=-1,
        non_position_lr_decay_until=-1, non_position_lr_final_mult=1.0)

def step(model, k):
    model.update_learning_rate(k)
    v = random.random() + np.random.rand()
    loss = sum((getattr(model, a) * torch.rand_like(getattr(model, a))).sum() * v
               for a in er.PARAMS.values())
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    model.xyz_gradient_accum.add_(1)
    model.denom.add_(2)
    return float(loss.item())

def example():
    opt = options()
    model = Model(opt)
    step(model, 16)
    m, o = er.capture_model(model)
    loop = {k: 0.1 for k in er.LOOP_FIELDS}
    loop.update(viewpoint_idx_stack=[], dense_viewpoint_idx_stack=[1, 0],
                chart_sampling_phase="early", use_chart_view_every_n_iter=5)
    static = {"chart_cameras": [{}, {}], "dense_cameras": [{}, {}]}
    binding = {"source": "fixed", "runtime": "CPU test", "iterations": 30000}
    state = {"schema": er.SCHEMA, "phase": er.PHASE, "completed_iteration": 16,
             "next_iteration": 17, "binding": binding, "static_ref": {"test": True}, "static_digest": "a" * 64,
             "model": m, "optimizer": o, "loop": loop, "rng": er.rng_state(),
             "events": {"evaluation": [], "save": [], "native_checkpoint": []},
             "pipeline": {"debug": False}, "scheduler_next": er.next_lrs(model, 17)}
    return model, opt, state, binding, static

class Contracts(unittest.TestCase):
    def setUp(self):
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)

    def test_exact_model_adam_rng_roundtrip_and_next_step(self):
        a, opt, state, binding, static = example()
        er.validate_checkpoint(state, binding, static)
        loss_a = step(a, 17)
        final_a = er.capture_model(a), er.rng_state()
        b = Model(opt)
        er.restore_model(b, state["model"], state["optimizer"], opt)
        self.assertEqual(er.next_lrs(b, 17), state["scheduler_next"])
        er.restore_rng(state["rng"])
        loss_b = step(b, 17)
        self.assertEqual(loss_a, loss_b)
        self.assertTrue(er.equal(final_a, (er.capture_model(b), er.rng_state())))

    def test_original_30000_scheduler_at_authentic_events(self):
        m = Model(options())
        for k in (1, 599, 600, 999, 1000, 2999, 3000, 3001, 7000, 7001, 14999, 15000, 15100, 30000):
            expected = er.next_lrs(m, k)
            m.update_learning_rate(k)
            self.assertEqual(expected, {g["name"]: g["lr"] for g in m.optimizer.param_groups})
        self.assertEqual(m.non_position_lr_decay_until, 30000)

    def test_atomic_readback_no_rng_consumption(self):
        _, _, state, _, _ = example()
        with tempfile.TemporaryDirectory() as root:
            before = er.rng_state()
            ref = er.atomic_save(Path(root) / "state", state)
            got = er.read_atomic(ref["path"], ref)
            self.assertTrue(er.equal(state, got))
            self.assertTrue(er.equal(before, er.rng_state()))
            with self.assertRaises(RuntimeError):
                er.atomic_save(ref["path"], state)

    def test_tensor_bitwise_stride_and_nan(self):
        x = torch.tensor([[float("nan"), -0.0], [1.0, 2.0]]).t()
        p = er.pack(x)
        self.assertTrue(er.equal(p, er.pack(er.unpack(p))))
        self.assertNotEqual(er.semantic_digest(torch.tensor([-0.])),
                            er.semantic_digest(torch.tensor([0.])))

    def test_none_empty_order_and_phase_are_preserved(self):
        _, _, state, binding, static = example()
        for stack in (None, [], [1, 0], [0, 1]):
            state["loop"]["viewpoint_idx_stack"] = stack
            er.validate_checkpoint(state, binding, static)
            self.assertTrue(er.equal(state["loop"], er.unpack(er.pack(state["loop"]))))
        state["loop"]["viewpoint_idx_stack"] = [0, 0]
        with self.assertRaises(RuntimeError):
            er.validate_checkpoint(state, binding, static)

    def test_native_sampling_refill_and_mutable_cadence_replay(self):
        fn = native_function(Path("/root/MAtCha/matcha/dm_scene/chart_sampling.py"),
                             "chart_regularization_indices_for_iteration", {})
        tree = ast.parse((SOURCE / "train_with_charts.py").read_text())
        native = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and
                      n.lineno == 1464)
        code = compile(ast.fix_missing_locations(ast.Module(body=[native], type_ignores=[])),
                       "native_sampling_block", "exec")
        env = {"randint": random.randint, "chart_regularization_indices_for_iteration": fn}
        cameras = [types.SimpleNamespace(image_name=str(i)) for i in range(3)]
        local = dict(iteration=0, use_chart_view_every_n_iter=5,
                     chart_regularization_indices=[2, 0], chart_late_regularization_indices=[1],
                     chart_late_prior_selection_from=3001, chart_sampling_phase=None,
                     viewpoint_idx_stack=None, dense_viewpoint_idx_stack=None,
                     viewpoint_cams=cameras, dense_viewpoint_cams=cameras)
        def run(values):
            records = []
            for k in values:
                local["iteration"] = k
                exec(code, env, local)
                records.append((local["viewpoint_idx"], local["chart_sampling_phase"],
                                copy.deepcopy(local["viewpoint_idx_stack"]),
                                copy.deepcopy(local["dense_viewpoint_idx_stack"])))
                if k > 3000 and k % local["use_chart_view_every_n_iter"] != 0:
                    local["use_chart_view_every_n_iter"] = 999999
            return records
        run(range(2990, 2999))
        saved, rng = copy.deepcopy(local), er.rng_state()
        a = run(range(2999, 3010))
        local.clear()
        local.update(saved)
        er.restore_rng(rng)
        self.assertEqual(a, run(range(2999, 3010)))
        self.assertEqual(local["use_chart_view_every_n_iter"], 999999)

    def test_reject_corrupt_or_incompatible_state_before_step(self):
        _, _, s, binding, static = example()
        mutations = [
            lambda x: x.update(schema="unknown"),
            lambda x: x.update(phase="before_optimizer"),
            lambda x: x.update(completed_iteration=30000, next_iteration=30001),
            lambda x: x.update(next_iteration=99),
            lambda x: x["binding"].update(runtime="changed"),
            lambda x: x["model"]["fields"].pop("mip_filter"),
            lambda x: x["loop"].pop("dense_viewpoint_idx_stack"),
            lambda x: x["model"]["bindings"].reverse(),
            lambda x: x["optimizer"]["param_groups"][0].update(name="unknown"),
            lambda x: x["model"]["fields"].pop("denom"),
            lambda x: x["optimizer"]["state"].pop(0),
            lambda x: x["optimizer"]["state"][0].pop("exp_avg"),
            lambda x: x["rng"].pop("numpy"),
            lambda x: x["events"]["save"].append(17000),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(s)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                er.validate_checkpoint(bad, binding, static)

    def test_reject_changed_config(self):
        m, opt, s, _, _ = example()
        opt.non_position_lr_decay_from = 100
        with self.assertRaises(RuntimeError):
            er.restore_model(m, s["model"], s["optimizer"], opt)

    def test_reject_gradient_and_wrong_parameter_binding(self):
        m, _, _, _, _ = example()
        m._xyz.grad = torch.zeros_like(m._xyz)
        with self.assertRaises(RuntimeError):
            er.capture_model(m)
        m._xyz.grad = None
        m.optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(m._xyz.detach().clone())
        with self.assertRaises(RuntimeError):
            er.capture_model(m)

    def test_reject_truncated_payload(self):
        _, _, state, _, _ = example()
        with tempfile.TemporaryDirectory() as root:
            ref = er.atomic_save(Path(root) / "state", state)
            p = Path(ref["path"]) / "state.pt"
            p.write_bytes(p.read_bytes()[:100])
            with self.assertRaises(RuntimeError):
                er.read_atomic(ref["path"])


    def test_controller_full_boundary_static_and_restore_micro(self):
        model, opt, example_state, _, _ = example()
        def fake_cameras():
            base = {k: 1 for k in er.CAMERA_FIELDS}
            base.update(image_name="a", original_image=torch.ones(3, 2, 2),
                        R=np.eye(3), T=np.zeros(3), data_device=torch.device("cpu"),
                        gt_alpha_mask=None)
            return [types.SimpleNamespace(**copy.deepcopy(base))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = types.SimpleNamespace(iterations=30000, use_color_correction=False,
                use_appearance_embedding=False, use_transient_model=False,
                start_checkpoint=None, early_stop_iteration=None, use_mip_filter=True,
                dense_data_path="cpu_fixture", dense_regul="default", exact_stop=16,
                exact_save=[], exact_windows=["16:17"], exact_dir=str(root / "baseline"),
                exact_resume=None, exact_manifest=str(root / "binding.json"), model_path="output")
            source_files = {str(Path(er.project_a_reference.__file__).resolve()): er.file_sha(er.project_a_reference.__file__),
                            str(Path(er.diagnostic_cpu.__file__).resolve()): er.file_sha(er.diagnostic_cpu.__file__),
                            str(Path(er.__file__).resolve()): er.file_sha(er.__file__),
                            str(Path(__file__).resolve()): er.file_sha(__file__)}
            source_files[str(Path(sys.argv[0]).resolve())] = er.file_sha(sys.argv[0])
            manifest = {"schema": "matcha_exact_binding_v1", "files": source_files,
                        "input_receipt_sha256": "a" * 64,
                        "science_args": {k: v for k, v in vars(args).items() if k not in er.CONTROL_ARGS}}
            Path(args.exact_manifest).write_text(json.dumps(manifest))
            def local(m):
                x = {k: torch.ones(1, 2, 2) for k in er.STATIC_FIELDS}
                x.update(chart_regularization_indices=[0], chart_late_regularization_indices=None,
                         chart_late_prior_weights=None, dense_depth_priors=[torch.ones(1, 2, 2)],
                         use_dense_supervision=True, use_depth_order_regularization=True,
                         charts_scale_factor=0.107, color_correction=None,
                         color_correction_optimizer=None, transient_model=None,
                         transient_optimizer=None, viewpoint_cams=fake_cameras(),
                         dense_viewpoint_cams=fake_cameras(), scene=types.SimpleNamespace(cameras_extent=1.),
                         rgb_mask_lookup=None, geometry_mask_lookup=None, gaussians=m, opt=opt,
                         pipe=types.SimpleNamespace(debug=False), iteration=16, testing_iterations=[],
                         saving_iterations=[], checkpoint_iterations=[], total_loss=torch.tensor(0.5))
                x.update({k: torch.tensor(0.5) for k in ("Ll1", "loss", "normal_loss", "dist_loss",
                    "depth_prior_loss", "normal_prior_loss", "curv_prior_loss", "total_regularization_loss")})
                x["rgb_losses"] = {"Ll1": torch.tensor(0.5), "loss": torch.tensor(0.5), "dssim": torch.tensor(0.5)}
                x.update(lambda_normal=0.05, lambda_dist=0., lambda_anisotropy=0.,
                         lambda_dense_depth=0., large_splat_radius_reg=0.)
                x.update(copy.deepcopy(example_state["loop"]))
                x["viewpoint_idx_stack"], x["dense_viewpoint_idx_stack"] = [], [0]
                x.update(viewpoint_idx=0, viewpoint_cam=x["dense_viewpoint_cams"][0])
                return x
            a = local(model)
            controller = er.Controller(args, {})
            self.assertIsNone(controller.prepare(a))
            controller.observe_view(a)
            rng = er.rng_state()
            self.assertTrue(controller.boundary(a))
            self.assertTrue(er.equal(rng, er.rng_state()))
            args.exact_resume = str(root / "baseline/step_00016")
            args.exact_dir = str(root / "branch")
            args.exact_stop = 17
            m2 = Model(opt)
            b = local(m2)
            b["dense_depth_priors"] = None  # Native inference is skipped on resume.
            restore = er.Controller(args, {})
            restored = restore.prepare(b)
            b.update(restored)
            self.assertTrue(er.equal(er.capture_model(model), er.capture_model(m2)))
            self.assertTrue(er.equal(a["dense_depth_priors"], b["dense_depth_priors"]))
            self.assertEqual(b["first_iter"], 16)
            self.assertEqual(b["viewpoint_idx_stack"], [])
            b["first_iter"] += 1
            restore.restore_rng_last()
            restore.verify_restored_boundary(b)
            self.assertTrue(er.equal(rng, er.rng_state()))
            restore.trace.close()

    def test_native_parser_science_recipe_without_gpu(self):
        import freeze_binding
        science = freeze_binding.parsed_science([str(SOURCE / "train_with_charts.py"),
                    "--iterations", "30000", "--use_mip_filter", "--dense_data_path", "dense",
                    "--normal_consistency_from", "7000", "--distortion_from", "3000"])
        self.assertEqual(science["iterations"], 30000)
        self.assertEqual(science["save_iterations"], [7000, 30000, 30000])
        self.assertEqual(science["non_position_lr_decay_until"], -1)
        self.assertFalse(science["use_appearance_embedding"])


    def test_stale_overlapping_file_binding_rejected(self):
        import freeze_binding
        files = {"native.py": "a" * 64}
        freeze_binding.add_bound_file(files, "native.py", "a" * 64)
        with self.assertRaises(AssertionError):
            freeze_binding.add_bound_file(files, "native.py", "b" * 64)
        self.assertEqual(files["native.py"], "a" * 64)

    def test_reference_coverage_fails_closed(self):
        records = {(k, full): {} for k in (600, 601, 602) for full in (False, True)}
        er.validate_reference(records, 600, 602, True)
        for key in ((600, False), (601, True), (602, True)):
            bad = dict(records)
            del bad[key]
            with self.assertRaises(RuntimeError):
                er.validate_reference(bad, 600, 602, True)
        with self.assertRaises(RuntimeError):
            er.validate_reference({}, 600, 602, True)
        with self.assertRaises(RuntimeError):
            er.validate_reference(records, 600, None, True)


    def test_generated_copies_match_fixed_hash_builders(self):
        import build_trainer
        import build_observer
        import contextlib
        import io
        for builder, filename in ((build_trainer, "train_with_charts_exact.py"),
                                  (build_observer, "train_with_charts_observer.py")):
            with tempfile.TemporaryDirectory() as root:
                old = builder.ROOT
                try:
                    builder.ROOT = Path(root)
                    with contextlib.redirect_stdout(io.StringIO()):
                        builder.main()
                    self.assertEqual((Path(root) / filename).read_text(),
                                     (Path(__file__).parent / filename).read_text())
                finally:
                    builder.ROOT = old


    def test_limited_gate_only_full_endpoints_required(self):
        from continuation_gate import ComparisonGate
        records = {(k, False): {"iteration": k, "fixed": "trace"} for k in range(15000, 15102)}
        records[(14999, True)] = {"exact_digest": "cut"}
        records[(15101, True)] = {"exact_digest": "end"}
        er.validate_reference(records, 15000, 30000, True, limited_gate=True)
        gate = ComparisonGate(records)
        with self.assertRaises(RuntimeError):
            gate.before_step(15102)
        gate.restored_boundary("cut")
        for k in range(15000, 15102):
            gate.observe(k, records[(k, False)], "end" if k == 15101 else None)
        self.assertTrue(gate.passed)
        gate.before_step(15102)
        for missing in ((15050, False), (14999, True), (15101, True)):
            bad = dict(records)
            del bad[missing]
            with self.assertRaises(RuntimeError):
                ComparisonGate(bad)
        gate = ComparisonGate(records)
        with self.assertRaises(RuntimeError):
            gate.restored_boundary("wrong")
        gate.restored_boundary("cut")
        with self.assertRaises(RuntimeError):
            gate.observe(15001, records[(15001, False)])

    def test_limited_gate_first_difference_missing_endpoint_fail(self):
        from continuation_gate import ComparisonGate
        records = {(k, False): {"iteration": k, "loss": "exact"} for k in range(15000, 15102)}
        records[(14999, True)] = {"exact_digest": "cut"}
        records[(15101, True)] = {"exact_digest": "end"}
        g = ComparisonGate(records)
        g.restored_boundary("cut")
        with self.assertRaises(RuntimeError):
            g.observe(15000, {"iteration": 15000, "loss": "different"})
        self.assertFalse(g.passed)
        g = ComparisonGate(records)
        g.restored_boundary("cut")
        for k in range(15000, 15101):
            g.observe(k, records[(k, False)])
        with self.assertRaises(RuntimeError):
            g.observe(15101, records[(15101, False)], None)
        self.assertFalse(g.passed)

    def test_missing_loss_component_rejected_and_signed_zero_exact(self):
        with self.assertRaises(RuntimeError):
            er.loss_components({"loss": torch.tensor(1.)}, "dense")
        self.assertNotEqual(er.scalar_record(torch.tensor(-0.)), er.scalar_record(torch.tensor(0.)))

    def test_n8_launcher_scope_uuid_output_and_deadline_negatives(self):
        import native_n8_launcher as launch
        spec = {"schema": "matcha_native_n8_launch_v1", "stage": "native_trial_a_n8",
            "physical_gpu": 0, "logical_gpu": 0, "visible_count": 1,
            "uuid": "GPU-00000000-0000-0000-0000-000000000001", "deadline_seconds": 3500,
            "forecast_seconds": 300, "output_root": str(launch.RUNS / "native_n8_trial_a_fixture"),
            "references": {k: {"path": "/cpu_fixture/" + k, "sha256": "a" * 64} for k in launch.REFS},
            "frozen_files": {str(launch.ROOT / p): "b" * 64 for p in launch.REQUIRED_FILES},
            "environment": {}}
        launch.validate_spec(spec)
        mutations = [lambda x: x.update(stage="baseline15101"), lambda x: x.update(physical_gpu=1),
            lambda x: x.update(logical_gpu=1), lambda x: x.update(visible_count=2),
            lambda x: x.update(deadline_seconds=3600), lambda x: x.update(forecast_seconds=3380),
            lambda x: x.update(output_root="/root/MAtCha"),
            lambda x: x["references"].pop("input_review"),
            lambda x: x["environment"].update(CUDA_VISIBLE_DEVICES="1")]
        for mutate in mutations:
            bad = copy.deepcopy(spec)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                launch.validate_spec(bad)
        launch.validate_context(spec, spec["uuid"], 1, 0)
        for actual, visible, logical in (("GPU-wrong", 1, 0), (spec["uuid"], 2, 0), (spec["uuid"], 1, 1)):
            with self.assertRaises(RuntimeError):
                launch.validate_context(spec, actual, visible, logical)
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "exclusive"
            launch.claim_output(out)
            with self.assertRaises(FileExistsError):
                launch.claim_output(out)
            p = Path(root) / "data"
            p.write_bytes(b"micro")
            with self.assertRaises(RuntimeError):
                launch.sha(p, deadline=0.)
        argv = launch.native_argv(spec, {"argv": ["native.py", "-m", "ignored", "--iterations", "30000"]}, "out")
        self.assertEqual(argv[argv.index("--exact_stop") + 1], "8")
        self.assertEqual(argv[argv.index("--iterations") + 1], "30000")

    def test_n8_cleanup_only_own_fake_group(self):
        import native_n8_launcher as launch
        calls = []
        p = types.SimpleNamespace(pid=987654321, wait=lambda timeout: 0)
        launch.cleanup_owned_group(p, killpg=lambda pid, sig: calls.append((pid, sig)))
        self.assertTrue(calls)
        self.assertTrue(all(pid == p.pid for pid, _ in calls))
        with self.assertRaises(RuntimeError):
            launch.cleanup_owned_group(types.SimpleNamespace(pid=os.getpgrp()),
                                       killpg=lambda *a: self.fail("must never kill parent"))


    def test_unique_cut_gate_budget_and_binding_rejections(self):
        import continuation_gate as cg
        value = {"schema": cg.SCHEMA, "checkpoint_iteration": 14999, "comparison_start": 15000,
            "comparison_end": 15101, "training_end": 30000, "binding_sha256": "a" * 64,
            "reference_trace_sha256": "b" * 64, "checkpoint14999_sha256": "c" * 64,
            "checkpoint15101_sha256": "d" * 64, "source_review_sha256": "e" * 64,
            "source_review_path": "/cpu_fixture/review.json",
            "budget": {"verdict": "PASS", "based_on_iteration": 14999,
                "measured_prefix_seconds": 2000., "measured_checkpoint_bytes": 4000,
                "measured_restore_seconds": 10., "predicted_total_seconds": 2200.,
                "deadline_seconds": 3500, "physical_gpu": 0, "logical_gpu": 0,
                "uuid": "GPU-00000000-0000-0000-0000-000000000001"}}
        cg.validate_gate(value, "a" * 64, "b" * 64, "c" * 64)
        mutations = [lambda x: x.update(comparison_end=15100), lambda x: x.update(training_end=15101),
            lambda x: x.update(checkpoint_iteration=15101),
            lambda x: x.update(reference_trace_sha256="f" * 64),
            lambda x: x["budget"].update(based_on_iteration=8),
            lambda x: x["budget"].update(predicted_total_seconds=3380),
            lambda x: x["budget"].update(measured_restore_seconds=0),
            lambda x: x["budget"].update(physical_gpu=2)]
        for mutate in mutations:
            bad = copy.deepcopy(value)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                cg.validate_gate(bad, "a" * 64, "b" * 64, "c" * 64)


    def test_first_difference_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            controller = er.Controller.__new__(er.Controller)
            controller.root = Path(root)
            controller.reference = {(15000, False): {"iteration": 15000, "loss": "original"}}
            for value in ("first_bad", "second_bad"):
                with self.assertRaises(RuntimeError):
                    controller.check_reference({"iteration": 15000, "loss": value}, (15000, False))
            report = json.loads((Path(root) / "first_difference.json").read_text())
            self.assertEqual(report["actual"]["loss"], "first_bad")

    def test_original_source_frozen_and_no_cuda(self):
        self.assertEqual(er.file_sha(SOURCE / "train_with_charts.py"),
                         "5c5c7d924675dc3ad743208b3e2ca35d3d064f30892338845ff2c08a30b3a630")
        self.assertEqual(er.file_sha(SOURCE / "scene/gaussian_model.py"),
                         "8e69f8adc84faf148f3ed3f9a1b82e8e9931373c1f98478852824df7adc514b4")
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(torch.cuda.device_count(), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
