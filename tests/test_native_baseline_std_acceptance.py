"""Synthetic CPU acceptance tests. Never create acceptance for a live model."""
import copy
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/native_baseline_repair"))
import accept_std as accept
import numpy as np
import torch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def checkpoint(path, iteration, identity):
    path.mkdir(parents=True)
    (path / "state.pt").write_bytes(b"synthetic checkpoint bytes")
    state = {key: None for key in sorted(accept.STATE_KEYS)}
    state.update(schema="std_native_exact_state_v1", iteration=iteration, identity=identity,
                 rng={"python": [1, 2, 3]})
    manifest = accept.primitive_manifest(state)
    digest = accept.manifest_hash(manifest)
    record = {"schema": "std_native_exact_state_v1", "complete": True, "iteration": iteration,
              "identity": identity, "blob": "state.pt", "blob_bytes": (path / "state.pt").stat().st_size,
              "blob_sha256": accept.sha(path / "state.pt"), "state_manifest": manifest, "state_digest": digest}
    write(path / "complete.json", record)
    write(path / "roundtrip_verified.json", {"complete": True, "iteration": iteration,
          "state_digest": digest, "blob_sha256": record["blob_sha256"], "roundtrip_location": "cpu",
          "original_storage_device_metadata_verified": True})
    return record, accept.manifest_hash(accept.primitive_manifest(state["rng"]))


def stage_fixture(root):
    semantic = "off"
    case = "STDLoc_StMarysChurch_mapping_semantics_off_train"
    model = root / "models" / case[:-6]
    (model / "detector").mkdir(parents=True)
    (model / "detector/30000_detector.pth").write_bytes(b"synthetic terminal product")
    command = ["python", "wrapper.py", "--receipt", "final.json", "--", "-m", str(model), "--iterations", "30000"]
    spec = {"fresh_model_path": str(model), "fresh_receipt_path": str(root / "receipts/final.json"),
            "new_job": {"command": command, "cwd": str(root / "native")}}
    protocol = {"source": str(root / "cambridge_data/StMarysChurch"), "jobs": {case: spec}, "native_code_sha256": {}}
    write(root / "training_protocol_v4.json", protocol)
    identity = {"protocol_sha256": accept.sha(root / "training_protocol_v4.json"),
                "build_sha256": "build", "mapping_semantics": semantic, "model_path": str(model),
                "source_alias": protocol["source"], "probe_only": False}
    build = {"file_sha256": {}, "generated_trainer": str(root / "generated.py")}
    previous = None
    for stage, start, stop in accept.STAGES:
        directory = root / "std_execution/production/off" / stage
        directory.mkdir(parents=True)
        native_command = list(command)
        native_command[native_command.index("--receipt") + 1] = str(directory / "native_receipt.json")
        config = {"stage": stage, "start": start, "stop": stop, "identity": identity, "probe_only": False,
                  "run_dir": str(directory), "runner_timeout_seconds": 3400,
                  "stage_receipt": str(directory / "native_receipt.json"), "final_receipt": spec["fresh_receipt_path"],
                  "wrapper_command": native_command, "resume": None if previous is None else str(previous[0]),
                  "code_sha256": {}, "generated_trainer": build["generated_trainer"],
                  "native_trainer": str(root / "native/train.py"), "checkpoint_at": -1 if stage == "detector" else stop,
                  "checkpoint_out": str(directory / ("checkpoint_" + str(stop))), "trace_start": 30001}
        write(directory / "config.json", config)
        launch = {"stage": stage, "build_sha256": "build", "runner_timeout_seconds": 3400,
                  "command": ["python", str(Path(accept.__file__).resolve().with_name("std_entry.py"))],
                  "nofile_before": [1024, 1048576], "nofile_after": [4096, 1048576]}
        write(directory / "launch.json", launch)
        terminal = {"case": case, "stage": stage, "mapping_semantics": semantic,
                    "terminal": True, "exit_code": 0, "evidence_source": "WebCodex observe_jobs",
                    "job_id": "wc_job_synthetic_" + stage, "observed_utc": "2026-10-12T10:00:00+00:00",
                    "config_sha256": accept.sha(directory / "config.json"),
                    "launch_sha256": accept.sha(directory / "launch.json")}
        write(directory / "terminal.json", terminal)
        native = {"status": "complete", "method": "STDLoc", "phase": "train", "mapping_semantics": "off",
                  "query_semantics": False, "seed": 2026, "wall_seconds": 100.0,
                  "build_stages": [{"stage": "training_detector", "wall_seconds": 90.0}] if stage == "detector" else []}
        write(directory / "native_receipt.json", native)
        done = dict(native, exact_stage=stage, native_total_iterations=30000,
                    gaussian_start=start, gaussian_stop=stop, probe_only=False,
                    protocol_sha256=identity["protocol_sha256"], build_sha256="build",
                    scope="single_stage_excludes_other_stages")
        write(directory / "stage_complete.json", done)
        if start:
            write(directory / "restore_verified.json", {"complete": True, "iteration": start,
                  "state_digest": previous[1]["state_digest"], "rng_digest": previous[2]})
        if stage != "detector":
            write(directory / "gaussian_complete.json", {"complete": True, "start": start, "stop": stop,
                  "native_total_iterations": 30000, "all_observed_native_losses_finite": True,
                  "continuously_floating_point_equivalent_trajectory_claimed": False,
                  "native_terminal_skips_optimizer_step": stop == 30000, "probe_only": False})
            write(directory / "trace.json", [{"iteration": i, "camera": "camera", "loss": 1.0,
                  "l1": 0.5, "feature_l1": 0.5, "optimizer_lr": [0.001]} for i in (start + 1, stop)])
            cp = directory / ("checkpoint_" + str(stop))
            record, rng = checkpoint(cp, stop, identity)
            previous = (cp, record, rng)
        else:
            write(directory / "detector_entry.json", {"complete": True, "iteration": 30000,
                  "rng_digest": previous[2], "natural_function": "train_detector.training_detector",
                  "train_iteration": 30000, "landmark_num": 16384, "landmark_k": 32})
            write(directory / "detector_complete.json", {"complete": True, "stage": "detector",
                  "train_iteration": 30000, "terminal_detector_tensors_finite": True,
                  "terminal_sha256": accept.sha(model / "detector/30000_detector.pth")})
        progress = "30000/30000" if stage == "detector" else str(stop-start) + "/" + str(30000-start)
        (directory / "native.log").write_text(progress + "\nTraining complete.\n")
    return protocol, build, identity


class AcceptanceTests(unittest.TestCase):
    def test_complete_four_stage_chain_and_progress_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, build, _ = stage_fixture(root)
            stages, evidence = accept.check_stage_chain(root, "off", protocol, "build", build)
            self.assertEqual(len(stages), 4)
            self.assertEqual(len(evidence), 4)
            self.assertFalse(torch.cuda.is_initialized())

    def test_missing_or_failed_official_terminal_rejected(self):
        for failure in ("missing", "exit", "config_hash", "job_reuse"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                protocol, build, _ = stage_fixture(root)
                path = root / "std_execution/production/off/gaussian_2/terminal.json"
                if failure == "missing":
                    path.unlink()
                else:
                    value = accept.read(path)
                    if failure == "exit":
                        value["exit_code"] = 1
                    elif failure == "config_hash":
                        value["config_sha256"] = "wrong"
                    else:
                        value["job_id"] = "wc_job_synthetic_gaussian_1"
                    write(path, value)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    accept.check_stage_chain(root, "off", protocol, "build", build)

    def test_boundary_roundtrip_restore_and_blob_drift_rejected(self):
        for failure in ("roundtrip", "restore", "blob", "manifest"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                protocol, build, _ = stage_fixture(root)
                base = root / "std_execution/production/off"
                if failure == "roundtrip":
                    (base / "gaussian_1/checkpoint_10000/roundtrip_verified.json").unlink()
                elif failure == "restore":
                    p = base / "gaussian_2/restore_verified.json"
                    value = accept.read(p); value["state_digest"] = "wrong"; write(p, value)
                elif failure == "blob":
                    (base / "gaussian_1/checkpoint_10000/state.pt").write_bytes(b"tampered")
                else:
                    p = base / "gaussian_1/checkpoint_10000/complete.json"
                    value = accept.read(p); value["state_manifest"]["dict"].pop(); write(p, value)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    accept.check_stage_chain(root, "off", protocol, "build", build)

    def test_native_command_and_detector_handoff_rejected(self):
        for failure in ("command", "detector", "probe"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                protocol, build, _ = stage_fixture(root)
                base = root / "std_execution/production/off"
                if failure == "detector":
                    path = base / "detector/detector_entry.json"; value = accept.read(path)
                    value["train_iteration"] = 1000
                else:
                    path = base / "gaussian_1/config.json"; value = accept.read(path)
                    if failure == "command":
                        value["wrapper_command"][-1] = "10000"
                    else:
                        value["probe_only"] = True
                write(path, value)
                with self.assertRaises(ValueError):
                    accept.check_stage_chain(root, "off", protocol, "build", build)

    def test_cpu_native_tensor_indices_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "indices.pkl"
            with path.open("wb") as stream:
                pickle.dump(torch.arange(16384), stream)
            value = accept.load_indices(path)
            self.assertEqual(accept.check_indices(value, 20000)["landmarks"], 16384)
            self.assertEqual(value.device.type, "cpu")
            with path.open("ab") as stream:
                stream.write(b"unexpected")
            with self.assertRaises(ValueError):
                accept.load_indices(path)

    def test_serialized_cuda_indices_are_mapped_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "indices.pkl"
            original = list(torch.serialization._package_registry)
            fake_cuda = (0, lambda storage: "cuda:0", lambda storage, location: None)
            with patch.object(torch.serialization, "_package_registry", [fake_cuda] + original):
                with path.open("wb") as stream:
                    pickle.dump(torch.arange(16384), stream)
            result = accept.load_indices(path)
            self.assertTrue(torch.equal(result, torch.arange(16384)))
            self.assertFalse(torch.cuda.is_initialized())

    def test_pickle_arbitrary_code_and_nested_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); marker = root / "bad"
            class Evil:
                def __reduce__(self):
                    return os.system, ("touch " + str(marker),)
            path = root / "evil.pkl"
            path.write_bytes(pickle.dumps(Evil()))
            with self.assertRaises(pickle.UnpicklingError):
                accept.load_indices(path)
            self.assertFalse(marker.exists())
            buffer = io.BytesIO(); torch.save(Evil(), buffer)
            payload = buffer.getvalue()
            class Nested:
                def __reduce__(self):
                    return torch.storage._load_from_bytes, (payload,)
            path.write_bytes(pickle.dumps(Nested()))
            with self.assertRaises(pickle.UnpicklingError):
                accept.load_indices(path)
            self.assertFalse(marker.exists())

    def test_invalid_indices_rejected(self):
        for value, count in ((torch.arange(10), 20000), (torch.zeros(16384, dtype=torch.int64), 20000),
                             (torch.arange(16384).float(), 20000), (torch.arange(16384), 10000)):
            with self.assertRaises(ValueError):
                accept.check_indices(value, count)

    def test_detector_full_shapes_finite_and_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.pth"
            state = {}
            for layer, inp, out in ((0, 256, 128), (2, 128, 64), (4, 64, 32), (6, 32, 1)):
                state["cnn." + str(layer) + ".weight"] = torch.zeros(out, inp, 3, 3)
                state["cnn." + str(layer) + ".bias"] = torch.zeros(out)
            torch.save(state, path)
            self.assertEqual(accept.check_detector(path)["detector_tensor_count"], 8)
            state["cnn.0.weight"][0, 0, 0, 0] = float("nan")
            torch.save(state, path)
            with self.assertRaises(ValueError):
                accept.check_detector(path)
            state.pop("cnn.0.weight")
            torch.save(state, path)
            with self.assertRaises(ValueError):
                accept.check_detector(path)

    def test_all_feature_gaussian_ply_fields_finite(self):
        from plyfile import PlyData, PlyElement
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.ply"
            vertices = np.zeros(16384, dtype=[(name, "f4") for name in accept.expected_ply_fields()])
            PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
            self.assertEqual(accept.check_ply(path)["ply_feature_dim"], 256)
            vertices["loc_255"][100] = float("inf")
            PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
            with self.assertRaisesRegex(ValueError, "loc_255"):
                accept.check_ply(path)

    def test_cfg_parse_never_executes_and_rejects_duplicate_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = root / "cfg_args"; marker = root / "bad"
            cfg.write_text("Namespace(x=1, y=False)")
            self.assertEqual(accept.parse_cfg(cfg), {"x": 1, "y": False})
            cfg.write_text("Namespace(x=__import__('os').system('touch " + str(marker) + "'))")
            with self.assertRaises(ValueError):
                accept.parse_cfg(cfg)
            self.assertFalse(marker.exists())
            cfg.write_text("Namespace(x=1, x=2)")
            with self.assertRaises(ValueError):
                accept.parse_cfg(cfg)

    def test_native_camera_split_order_pose_and_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = ["train_%04d.png" % i for i in range(1487)]
            test = ["test_%04d.png" % i for i in range(530)]
            (root / "dataset_train.txt").write_text("# header\n# header\n# header\n" + "\n".join(train))
            (root / "dataset_test.txt").write_text("# header\n# header\n# header\n" + "\n".join(test))
            names = sorted(test) + sorted(train)
            extrinsics = {i: SimpleNamespace(name=name, camera_id=1, qvec=np.array([1.,0,0,0]),
                                            tvec=np.array([1.,2.,3.])) for i, name in enumerate(names)}
            intrinsics = {1: SimpleNamespace(width=640, height=480, model="PINHOLE",
                                            params=np.array([1000.,950.,320.,240.]))}
            readers = (lambda path: intrinsics, lambda path: extrinsics, lambda q: np.eye(3))
            rows = [{"id": i, "img_name": name, "width": 640, "height": 480,
                     "position": [-1.,-2.,-3.], "rotation": np.eye(3).tolist(), "fx":1000., "fy":950.}
                    for i, name in enumerate(names)]
            path = root / "cameras.json"; write(path, rows)
            result = accept.check_cameras(path, root, readers)
            self.assertEqual(result["camera_count"], 2017)
            rows[0]["position"][0] = 1.0; write(path, rows)
            with self.assertRaisesRegex(ValueError, "pose"):
                accept.check_cameras(path, root, readers)

    def test_trace_requires_real_stage_bounds_and_finite(self):
        rows = [{"iteration": i, "camera": "cam", "loss": 1.0, "l1": 0.5,
                 "feature_l1": 0.5, "optimizer_lr": [0.001]} for i in (10001, 20000)]
        self.assertEqual(accept.check_trace(rows, 10000, 20000), 2)
        rows[-1]["loss"] = float("nan")
        with self.assertRaises(ValueError):
            accept.check_trace(rows, 10000, 20000)

    def test_terminal_bool_exit_code_and_no_timezone_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, build, _ = stage_fixture(root)
            directory = root / "std_execution/production/off/gaussian_1"
            value = accept.read(directory / "terminal.json")
            for key, bad in (("exit_code", False), ("observed_utc", "2026-10-12T10:00:00"), ("evidence_source", "self report")):
                changed = dict(value, **{key: bad})
                with self.assertRaises(ValueError):
                    accept.check_terminal(changed, value["case"], "off", "gaussian_1",
                                          directory / "config.json", directory / "launch.json")

    def test_atomic_acceptance_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acceptance.json"
            accept.atomic_new(path, {"complete": True})
            with self.assertRaises(FileExistsError):
                accept.atomic_new(path, {"complete": False})
            self.assertTrue(accept.read(path)["complete"])

    @unittest.skipIf(sys.version_info < (3, 9), "Existing query gate uses Python3.9+ str.removesuffix; tested unmodified under g4splat")
    def test_query_gate_acceptance_shape_compatible(self):
        import query_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root / "model"; model.mkdir()
            protocol = root / "protocol.json"; protocol.write_text("{}")
            assets = {}
            names = {"cfg_args": "cfg_args", "point_cloud/iteration_30000/point_cloud.ply": "ply",
                     "detector/sampled_idx.pkl": "indices", "detector/30000_detector.pth": "detector"}
            artifacts = {}
            for relative, name in names.items():
                path = model / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"synthetic")
                assets[relative] = accept.ref(path); artifacts[name] = assets[relative]
            case = "STDLoc_StMarysChurch_mapping_semantics_off_eval"
            value = {"complete": True, "case": case[:-5] + "_train",
                     "method": "STDLoc", "semantics": "off", "seed": 2026, "iterations": 30000,
                     "training_protocol": accept.ref(protocol), "artifacts": artifacts}
            query_plan.verify_accepted_model(value, case, "STDLoc", "off", model, assets, accept.ref(protocol))
            value["seed"] = 2025
            with self.assertRaises(ValueError):
                query_plan.verify_accepted_model(value, case, "STDLoc", "off", model, assets, accept.ref(protocol))


if __name__ == "__main__":
    unittest.main()
