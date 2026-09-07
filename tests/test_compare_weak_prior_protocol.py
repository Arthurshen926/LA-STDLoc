from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.compare_weak_prior_protocol import check_manifest_pair, compare


def _manifest(root: str, prior: str) -> dict:
    names = ["observations", "v2_audit_000", "projective_map", "base_evaluation"]
    return {
        "schema": "anygsloc_scene_experiment", "complete": True,
        "scene": "scene", "dataset": "/data", "config": {},
        "scientific_scope": {"mapping_uses_source_rgb": False},
        "mapping_detection": {"raw_keypoints": 2048},
        "prior": {"ply": prior, "gaussian_type": "2dgs", "sh_degree": 3,
                  "white_background": False},
        "stages": [{"name": n, "command": ["python", prior, root + "/" + n]}
                   for n in names],
        "results": {n: {"status": "executed"} for n in names},
    }


def test_allows_only_prior_and_output_path_changes() -> None:
    check_manifest_pair(_manifest("/a", "/p1"), _manifest("/b", "/p2"),
                        (Path("/a"), Path("/b")))


@pytest.mark.parametrize("change", ["reuse", "threshold", "scope", "unfinished"])
def test_rejects_incomplete_or_confounded_protocol(change: str) -> None:
    left = _manifest("/a", "/p1")
    right = deepcopy(_manifest("/b", "/p2"))
    if change == "reuse":
        right["results"]["observations"]["status"] = "reused"
    elif change == "threshold":
        right["stages"][0]["command"].extend(["--threshold", "0.1"])
    elif change == "scope":
        right["scientific_scope"]["mapping_uses_source_rgb"] = True
    else:
        right["complete"] = False
    with pytest.raises(ValueError):
        check_manifest_pair(left, right, (Path("/a"), Path("/b")))


def test_reads_source_identity_from_map_provenance_and_checks_artifact_bytes(tmp_path) -> None:
    import torch
    from common.hashing import sha256_file

    roots = (tmp_path / "left", tmp_path / "right")
    for index, root in enumerate(roots):
        root.mkdir()
        mapping = root / "projective_map"
        mapping.mkdir()
        evaluation = root / "base_evaluation"
        evaluation.mkdir()
        manifest = _manifest(str(root), f"/prior{index}")
        manifest["stages"][-1]["output"] = str(evaluation / "summary.json")
        (root / "anygsloc_scene_manifest.json").write_text(json.dumps(manifest))
        path = mapping / "projective_anchor_map.pt"
        torch.save({"provenance": {"implementation_sha256": {"source": "same"}},
                    "v6_mapping_query_names": ["mapping"]}, path)
        (mapping / "report.json").write_text(json.dumps({
            "contracts": {"policy": "same", "projective_completion_available": bool(index)},
            "counts": {"mapping_views": 1},
            "output": {"map_sha256": sha256_file(path)},
        }))
        summary = {key: 1.0 for key in (
            "median_te_cm", "mean_te_cm", "p90_te_cm", "median_ae_deg", "mean_ae_deg",
            "p90_ae_deg", "catastrophic_100cm_count", "matching_ms_mean",
            "ransac_ms_mean", "total_ms_mean", "query_count", "random_seed",
            "ransac_confidence", "ransac_minimum_iterations", "ransac_maximum_iterations")}
        summary["evaluated_split"] = "test"
        (evaluation / "summary.json").write_text(json.dumps(summary))
        (evaluation / "results.json").write_text(json.dumps([{"image_name": "query"}]))
    assert compare(roots)["status"] == "COMMAND_AND_REPORT_PARITY_PASS"
    # Geometry may differ across priors; bytes must still match their own report.
    torch.save({}, roots[0] / "projective_map/projective_anchor_map.pt")
    with pytest.raises(ValueError, match="map bytes"):
        compare(roots)
