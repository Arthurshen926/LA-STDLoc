import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_stdloc_eval_cfg.py"


class MakeStdlocEvalCfgTest(unittest.TestCase):
    def test_preserves_base_artifact_binding_without_detector_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_cfg = tmp / "base.yaml"
            out_cfg = tmp / "out.yaml"
            base_cfg.write_text(
                yaml.dump(
                    {
                        "sparse": {
                            "detector_path": "ulfloc_native_no_detector/0_detector.pth",
                            "landmark_path": "/maps/oldhospital/sampled_idx.pkl",
                            "landmark_meta_path": "/maps/oldhospital/landmark_meta.pt",
                        },
                        "dense": {},
                    }
                )
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base_cfg",
                    str(base_cfg),
                    "--output",
                    str(out_cfg),
                    "--artifact_model_path",
                    "/tmp/model",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            sparse = yaml.load(
                out_cfg.read_text(), Loader=yaml.FullLoader
            )["sparse"]
            self.assertEqual(
                sparse["detector_path"], "ulfloc_native_no_detector/0_detector.pth"
            )
            self.assertEqual(
                sparse["landmark_path"], "/maps/oldhospital/sampled_idx.pkl"
            )
            self.assertEqual(
                sparse["landmark_meta_path"], "/maps/oldhospital/landmark_meta.pt"
            )

    def test_preserves_all_unspecified_sparse_runtime_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_cfg = tmp / "base.yaml"
            out_cfg = tmp / "out.yaml"
            preserved = {
                "unique_landmark_matches": True,
                "use_candidate_dustbin": True,
                "pair_scorer_threshold": 0.37,
                "use_pair_measurement": True,
                "use_pair_measurement_offset": False,
                "pair_measurement_refill_mode": "geometry",
                "min_candidate_matches": 777,
                "use_detector_offset": True,
                "full_primitive_retrieval": True,
                "full_primitive_chunk_size": 4096,
                "diagnostics": {
                    "enabled": False,
                    "grid_rows": 9,
                    "dump_pre_selector": False,
                },
                "geometry_balance": {"enabled": True, "max_matches": 321},
            }
            base_cfg.write_text(yaml.dump({"sparse": preserved, "dense": {}}))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base_cfg",
                    str(base_cfg),
                    "--output",
                    str(out_cfg),
                    "--artifact_model_path",
                    "/tmp/model",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            sparse = yaml.load(out_cfg.read_text(), Loader=yaml.FullLoader)["sparse"]
            for key, value in preserved.items():
                self.assertEqual(sparse[key], value)

    def test_writes_native_matchability_solver_only_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_cfg = tmp / "base.yaml"
            out_cfg = tmp / "out.yaml"
            base_cfg.write_text(yaml.dump({"sparse": {}, "dense": {}}))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base_cfg", str(base_cfg),
                    "--output", str(out_cfg),
                    "--artifact_model_path", "/tmp/model",
                    "--use_native_matchability",
                    "--native_matchability_state_path", "/tmp/calibrator.pt",
                    "--native_matchability_max_prosac_iterations", "40000",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            sparse = yaml.load(out_cfg.read_text(), Loader=yaml.FullLoader)["sparse"]
            self.assertTrue(sparse["use_native_matchability"])
            self.assertEqual(sparse["native_matchability_state_path"], "/tmp/calibrator.pt")
            self.assertEqual(sparse["native_matchability_max_prosac_iterations"], 40000)

    def test_preserves_unspecified_matching_caps_and_frontend_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_cfg = tmp / "base.yaml"
            out_cfg = tmp / "out.yaml"
            base_cfg.write_text(
                yaml.dump(
                    {
                        "sparse": {
                            "max_matches_per_keypoint": 1,
                            "max_matches_per_landmark": 2,
                            "candidate_frontend_match_policy": "ignore",
                        },
                        "dense": {},
                    }
                )
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base_cfg",
                    str(base_cfg),
                    "--output",
                    str(out_cfg),
                    "--artifact_model_path",
                    "/tmp/model",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            sparse = yaml.load(
                out_cfg.read_text(), Loader=yaml.FullLoader
            )["sparse"]
            self.assertEqual(sparse["max_matches_per_keypoint"], 1)
            self.assertEqual(sparse["max_matches_per_landmark"], 2)
            self.assertEqual(
                sparse["candidate_frontend_match_policy"], "ignore"
            )

    def test_writes_detector_landmark_artifact_paths_and_sparse_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_cfg = tmp / "base.yaml"
            out_cfg = tmp / "out.yaml"
            base_cfg.write_text(
                yaml.dump(
                    {
                        "sparse": {
                            "detect_num": 2048,
                            "reprojection_error": 12.0,
                            "detector_path": "detector/30000_detector.pth",
                        },
                        "dense": {},
                    }
                )
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base_cfg",
                    str(base_cfg),
                    "--output",
                    str(out_cfg),
                    "--artifact_model_path",
                    "/tmp/model",
                    "--detector_folder",
                    "detector_la",
                    "--detector_iters",
                    "123",
                    "--candidate_teacher_state_path",
                    "detector_la/candidate_teacher_state.pt",
                    "--pair_scorer_state_path",
                    "detector_la/pair_scorer_state.pt",
                    "--landmark_feature_override_path",
                    "detector_la/map_feature_state.pt",
                    "--override_landmark_features",
                    "--detect_num",
                    "4096",
                    "--reprojection_error",
                    "8.0",
                    "--nms",
                    "2",
                    "--match_threshold",
                    "0.65",
                    "--match_topk",
                    "4",
                    "--unique_landmark_matches",
                    "--max_matches_per_landmark",
                    "3",
                    "--max_matches_per_keypoint",
                    "1",
                    "--use_candidate_dustbin",
                    "--use_candidate_pair_scorer",
                    "--pair_scorer_threshold",
                    "0.2",
                    "--use_candidate_pair_scorer_calibrated_threshold",
                    "--min_candidate_matches",
                    "1200",
                    "--candidate_refill_trigger_count",
                    "800",
                    "--use_detector_matchability",
                    "--detector_matchability_mode",
                    "proposal_rerank",
                    "--use_detector_offset",
                    "--detector_max_offset",
                    "1.5",
                    "--candidate_frontend_match_policy",
                    "error",
                    "--geometry_balance",
                    "--geometry_balance_max_per_cell",
                    "8",
                    "--geometry_balance_max_matches",
                    "512",
                    "--diagnostics_grid_rows",
                    "3",
                    "--diagnostics_grid_cols",
                    "5",
                    "--diagnostics_voxel_size",
                    "0.5",
                    "--diagnostics_dump_correspondences",
                    "--diagnostics_dump_all",
                    "--diagnostics_dump_discrete_oracle",
                    "--diagnostics_oracle_topk",
                    "16",
                    "--summary_json",
                    str(tmp / "summary.json"),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            cfg = yaml.load(out_cfg.read_text(), Loader=yaml.FullLoader)
            self.assertEqual(cfg["model_path"], "/tmp/model")
            sparse = cfg["sparse"]
            self.assertEqual(sparse["detector_path"], "detector_la/123_detector.pth")
            self.assertEqual(sparse["landmark_path"], "detector_la/sampled_idx.pkl")
            self.assertEqual(sparse["landmark_meta_path"], "detector_la/landmark_meta.pt")
            self.assertEqual(sparse["detector_model_path"], "/tmp/model")
            self.assertEqual(sparse["landmark_model_path"], "/tmp/model")
            self.assertEqual(sparse["landmark_meta_model_path"], "/tmp/model")
            self.assertEqual(
                sparse["candidate_teacher_state_path"],
                "detector_la/candidate_teacher_state.pt",
            )
            self.assertEqual(sparse["candidate_teacher_state_model_path"], "/tmp/model")
            self.assertEqual(
                sparse["pair_scorer_state_path"],
                "detector_la/pair_scorer_state.pt",
            )
            self.assertEqual(sparse["pair_scorer_state_model_path"], "/tmp/model")
            self.assertEqual(
                sparse["landmark_feature_override_path"],
                "detector_la/map_feature_state.pt",
            )
            self.assertEqual(
                sparse["landmark_feature_override_model_path"], "/tmp/model"
            )
            self.assertTrue(sparse["override_landmark_features"])
            self.assertEqual(sparse["detect_num"], 4096)
            self.assertEqual(float(sparse["reprojection_error"]), 8.0)
            self.assertEqual(sparse["nms"], 2)
            self.assertEqual(float(sparse["threshold"]), 0.65)
            self.assertEqual(sparse["topk"], 4)
            self.assertTrue(sparse["unique_landmark_matches"])
            self.assertEqual(sparse["max_matches_per_landmark"], 3)
            self.assertEqual(sparse["max_matches_per_keypoint"], 1)
            self.assertTrue(sparse["use_candidate_dustbin"])
            self.assertTrue(sparse["use_candidate_pair_scorer"])
            self.assertEqual(float(sparse["pair_scorer_threshold"]), 0.2)
            self.assertTrue(
                sparse["use_candidate_pair_scorer_calibrated_threshold"]
            )
            self.assertEqual(sparse["min_candidate_matches"], 1200)
            self.assertEqual(sparse["candidate_refill_trigger_count"], 800)
            self.assertTrue(sparse["use_detector_matchability"])
            self.assertEqual(sparse["detector_matchability_mode"], "proposal_rerank")
            self.assertTrue(sparse["use_detector_offset"])
            self.assertEqual(float(sparse["detector_max_offset"]), 1.5)
            self.assertEqual(sparse["candidate_frontend_match_policy"], "error")
            self.assertFalse(sparse["use_landmark_prior"])
            self.assertTrue(sparse["diagnostics"]["enabled"])
            self.assertTrue(sparse["diagnostics"]["gt_metrics"])
            self.assertTrue(sparse["diagnostics"]["dump_correspondences"])
            self.assertFalse(sparse["diagnostics"]["dump_inliers_only"])
            self.assertTrue(sparse["diagnostics"]["dump_pre_selector"])
            self.assertTrue(sparse["diagnostics"]["dump_discrete_oracle"])
            self.assertEqual(sparse["diagnostics"]["oracle_topk"], 16)
            self.assertEqual(sparse["diagnostics"]["grid_rows"], 3)
            self.assertEqual(sparse["diagnostics"]["grid_cols"], 5)
            self.assertEqual(float(sparse["diagnostics"]["voxel_size"]), 0.5)
            self.assertTrue(sparse["geometry_balance"]["enabled"])
            self.assertEqual(sparse["geometry_balance"]["max_per_cell"], 8)
            self.assertEqual(sparse["geometry_balance"]["max_matches"], 512)

            summary = json.loads((tmp / "summary.json").read_text())
            self.assertEqual(summary["output"], str(out_cfg))
            self.assertEqual(summary["detector_path"], "detector_la/123_detector.pth")
            self.assertEqual(
                summary["candidate_teacher_state_path"],
                "detector_la/candidate_teacher_state.pt",
            )
            self.assertEqual(
                summary["pair_scorer_state_path"],
                "detector_la/pair_scorer_state.pt",
            )
            self.assertEqual(
                summary["landmark_feature_override_path"],
                "detector_la/map_feature_state.pt",
            )
            self.assertTrue(summary["override_landmark_features"])
            self.assertEqual(summary["nms"], 2)
            self.assertEqual(float(summary["match_threshold"]), 0.65)
            self.assertEqual(summary["match_topk"], 4)
            self.assertTrue(summary["unique_landmark_matches"])
            self.assertEqual(summary["max_matches_per_landmark"], 3)
            self.assertEqual(summary["max_matches_per_keypoint"], 1)
            self.assertTrue(summary["use_candidate_dustbin"])
            self.assertTrue(summary["use_candidate_pair_scorer"])
            self.assertEqual(float(summary["pair_scorer_threshold"]), 0.2)
            self.assertTrue(
                summary["use_candidate_pair_scorer_calibrated_threshold"]
            )
            self.assertEqual(summary["min_candidate_matches"], 1200)
            self.assertEqual(summary["candidate_refill_trigger_count"], 800)
            self.assertTrue(summary["use_detector_matchability"])
            self.assertEqual(summary["detector_matchability_mode"], "proposal_rerank")
            self.assertTrue(summary["use_detector_offset"])
            self.assertEqual(float(summary["detector_max_offset"]), 1.5)
            self.assertEqual(summary["candidate_frontend_match_policy"], "error")
            self.assertTrue(summary["diagnostics"]["enabled"])
            self.assertEqual(summary["diagnostics"]["grid_rows"], 3)
            self.assertEqual(summary["diagnostics"]["grid_cols"], 5)
            self.assertEqual(float(summary["diagnostics"]["voxel_size"]), 0.5)
            self.assertTrue(summary["geometry_balance"]["enabled"])

    def test_allows_detector_and_landmark_artifacts_to_be_decoupled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_cfg = tmp / "base.yaml"
            out_cfg = tmp / "out.yaml"
            base_cfg.write_text(yaml.dump({"sparse": {}, "dense": {}}))

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base_cfg",
                    str(base_cfg),
                    "--output",
                    str(out_cfg),
                    "--artifact_model_path",
                    "/tmp/model",
                    "--detector_folder",
                    "frozen_detector",
                    "--detector_iters",
                    "2000",
                    "--landmark_path",
                    "/tmp/map/sampled_idx.pkl",
                    "--landmark_meta_path",
                    "/tmp/map/landmark_meta.pt",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            sparse = yaml.load(
                out_cfg.read_text(), Loader=yaml.FullLoader
            )["sparse"]
            self.assertEqual(
                sparse["detector_path"], "frozen_detector/2000_detector.pth"
            )
            self.assertEqual(
                sparse["landmark_path"], "/tmp/map/sampled_idx.pkl"
            )
            self.assertEqual(
                sparse["landmark_meta_path"], "/tmp/map/landmark_meta.pt"
            )

    def test_writes_explicit_ulfloc_native_frontend(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_cfg = tmp / "base.yaml"
            out_cfg = tmp / "out.yaml"
            base_cfg.write_text(yaml.dump({"sparse": {}, "dense": {}}))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base_cfg",
                    str(base_cfg),
                    "--output",
                    str(out_cfg),
                    "--artifact_model_path",
                    "/tmp/model",
                    "--sparse_query_feature_contract",
                    "native_resized_input",
                    "--sparse_frontend",
                    "ulfloc_native",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            sparse = yaml.load(out_cfg.read_text(), Loader=yaml.FullLoader)["sparse"]
            self.assertEqual(sparse["sparse_frontend"], "ulfloc_native")
            self.assertEqual(
                sparse["query_feature_contract"], "native_resized_input"
            )


if __name__ == "__main__":
    unittest.main()
