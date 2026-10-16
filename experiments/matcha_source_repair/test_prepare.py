"""Small CPU-only contract tests. Uses synthetic data, never real source assets."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from plyfile import PlyData, PlyElement

spec = importlib.util.spec_from_file_location("repair", Path(__file__).with_name("prepare.py"))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

class TransportContract(unittest.TestCase):
    def setUp(self):
        self.old = np.eye(4)
        self.old[:3, :3] = [[1.00003, .00002, 0], [-.00001, .99998, 0], [0, 0, 1]]
        self.old[:3, 3] = [4, -3, 10]
        self.new = np.eye(4)
        self.new[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.new[:3, 3] = [2, 5, -4]
        self.local = np.array([[1., 2., 3.], [-4, 6, 8]])
        self.world = self.local @ self.old[:3, :3].T + self.old[:3, 3]

    def test_affine_local_frame(self):
        new, local = repair.transport(self.world, self.old, self.new)
        np.testing.assert_allclose(local, self.local, rtol=0, atol=1e-12)
        np.testing.assert_allclose(new @ self.new[:3, :3].T + self.new[:3, 3],
                                   self.local, rtol=0, atol=1e-12)

    def test_scaled_float32_roundtrip(self):
        s = .10705874667395594
        old_pts = (self.world * s).astype(np.float32)
        new, local = repair.transport(old_pts.astype(float) / s, self.old, self.new)
        restored = (new * s).astype(np.float32).astype(float) / s
        np.testing.assert_allclose(restored @ self.new[:3, :3].T + self.new[:3, 3],
                                   local, rtol=3e-7, atol=5e-5)

    def test_bad_transforms_rejected(self):
        bad = self.old.copy(); bad[3, 0] = .1
        with self.assertRaisesRegex(ValueError, "homogeneous"):
            repair.transport(self.world, bad, self.new)
        bad = self.old.copy(); bad[:3, :3] = 0
        with self.assertRaisesRegex(ValueError, "singular"):
            repair.transport(self.world, bad, self.new)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            repair.transport([[float("nan"), 0, 0]], self.old, self.new)

class BinaryContract(unittest.TestCase):
    def blob(self):
        record = struct.pack("<idddddddi", 7, 1.0001, 0, 0, 0, 1, 2, 3, 4)
        return struct.pack("<Q", 1) + record + b"seq__frame.png\0" + struct.pack("<Qddq", 1, 2, 3, 99)

    def test_only_qt_replaced(self):
        blob = self.blob()
        new = repair.patch_image_bytes(blob, {"seq__frame.png": ([1., 0, 0, 0], [4., 5, 6])})
        self.assertEqual(blob[:12], new[:12])
        self.assertEqual(blob[68:], new[68:])
        self.assertEqual(struct.unpack_from("<7d", new, 12), (1., 0., 0., 0., 4., 5., 6.))

    def test_missing_name_and_bad_q_fail(self):
        with self.assertRaisesRegex(ValueError, "name-set"):
            repair.patch_image_bytes(self.blob(), {})
        with self.assertRaisesRegex(ValueError, "nonunit"):
            repair.patch_image_bytes(self.blob(), {"seq__frame.png": ([2., 0, 0, 0], [0, 0, 0])})
        with self.assertRaisesRegex(ValueError, "trailing"):
            repair.image_records(self.blob() + b"x")

    def test_point_owner_transport_and_nonxyz_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / "src", Path(temp) / "dst"
            for p in (src, dst): (p / "sparse/0").mkdir(parents=True)
            points = np.zeros(2, dtype=repair.POINT_DTYPE)
            points["id"], points["x"], points["y"], points["z"] = [1, 2], [1, 2], [3, 4], [5, 6]
            points["track_length"], points["image_id"], points["r"] = 1, [7, 8], [22, 33]
            raw = struct.pack("<Q", 2) + points.tobytes()
            (src / "sparse/0/points3D.bin").write_bytes(raw)
            vertex = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1")])
            for k in ("x", "y", "z"): vertex[k] = points[k]
            vertex["red"] = points["r"]
            PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(src / "sparse/0/points3D.ply")
            old = {7: np.eye(4), 8: np.eye(4)}
            new = {7: np.eye(4), 8: np.eye(4)}
            new[7][0, 3] = 10; new[8][1, 3] = -20
            report = repair.transform_sparse_points(src, dst, old, new)
            self.assertEqual(report["count"], 2)
            out = np.frombuffer((dst / "sparse/0/points3D.bin").read_bytes(), dtype=repair.POINT_DTYPE, offset=8)
            np.testing.assert_array_equal(out["x"], [-9, 2])
            np.testing.assert_array_equal(out["y"], [3, 24])
            for k in set(repair.POINT_DTYPE.names) - {"x", "y", "z"}:
                np.testing.assert_array_equal(out[k], points[k])
            broken = points.copy(); broken["track_length"][0] = 2
            (src / "sparse/0/points3D.bin").write_bytes(struct.pack("<Q", 2) + broken.tobytes())
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                repair.transform_sparse_points(src, dst, old, new)

class IntrinsicsContract(unittest.TestCase):
    def setup_inputs(self):
        from types import SimpleNamespace as S
        bindings = {"seq__frame.png": 4}
        cameras = {4: S(id=4, model="PINHOLE", width=1920, height=1080,
                        params=np.array([1669., 1669., 960., 540.]))}
        corrected = {"seq/frame.png": S(camera_id=9, qvec=np.array([1., 0, 0, 0]), tvec=np.zeros(3))}
        reference = {9: S(id=9, model="SIMPLE_RADIAL", width=1920, height=1080,
                         params=np.array([1669., 960., 540., .045]))}
        return bindings, cameras, corrected, reference

    def test_dense_name_binding_and_pinhole_projection(self):
        args = self.setup_inputs()
        report = repair.validate_dense_intrinsics(*args)
        self.assertEqual(report["checked_count"], 1)
        self.assertEqual(report["per_name"][0]["corrected_radial_k"], .045)
        repair.validate_copied_intrinsics(args[1], args[1])

    def test_same_pose_changed_dense_fx_rejected(self):
        args = self.setup_inputs()
        args[1][4].params[0] += 1
        with self.assertRaisesRegex(ValueError, "pinhole K"):
            repair.validate_dense_intrinsics(*args)

    def test_same_pose_changed_dense_size_rejected(self):
        for dimension in ("width", "height"):
            args = self.setup_inputs()
            setattr(args[1][4], dimension, getattr(args[1][4], dimension) + 1)
            with self.assertRaisesRegex(ValueError, "image size"):
                repair.validate_dense_intrinsics(*args)

    def test_model_or_camera_id_binding_rejected(self):
        args = self.setup_inputs(); args[1][4].model = "SIMPLE_PINHOLE"
        with self.assertRaisesRegex(ValueError, "model"):
            repair.validate_dense_intrinsics(*args)
        args = self.setup_inputs(); args[2]["seq/frame.png"].camera_id = 99
        with self.assertRaisesRegex(ValueError, "camera_id"):
            repair.validate_dense_intrinsics(*args)

    def test_copied_chart_intrinsics_change_rejected(self):
        import copy
        original = self.setup_inputs()[1]
        changed = copy.deepcopy(original); changed[4].params[0] += 1
        with self.assertRaisesRegex(ValueError, "copied camera"):
            repair.validate_copied_intrinsics(original, changed)

class RecipeContract(unittest.TestCase):
    def test_frozen_horizon_and_type(self):
        p = repair.frozen_recipe(Path("/example"), Path("/chart"), Path("/dense"))
        self.assertEqual(p["argv"][p["argv"].index("--iterations") + 1], "30000")
        self.assertEqual(p["seed"], 0)
        self.assertEqual(p["type"], "2dgs")
        self.assertFalse(p["ready_for_gpu"])

    def test_unauthorized_destination_fails_before_reads(self):
        with self.assertRaisesRegex(ValueError, "authorized isolated"):
            repair.prepare(Path("/tmp/not-authorized-matcha-output"))

if __name__ == "__main__":
    unittest.main(verbosity=2)
