import json
from pathlib import Path

import pytest
import torch

from common.v7_contracts import (
    audit_formal_import_graph,
    compare_query_results,
    load_v7_config,
    tensor_tree_equal,
    validate_compact_map,
)


def _map() -> dict:
    return {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.tensor([3, 7]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]]),
        "anchor_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "projective_anchor_construction": {
            "final_xyz_source": "fixed_camera_robust_ray_triangulation",
            "gaussian_depth_role": "proposal_and_visibility_only",
            "direct_gaussian_surface_anchor": False,
        },
        "provenance": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "uses_gaussian_geometry_for_triangulation": False,
        },
    }


def test_checked_in_v7_config_matches_immutable_contract() -> None:
    payload = load_v7_config("configs/v7_safe_closed_loop.yaml")
    assert payload["future_phases"]["p3_unified_selector"] == "disabled_until_p2_passes"


def test_v7_map_contract_is_one_anchor_one_descriptor_and_pure_ray() -> None:
    assert validate_compact_map(_map()) == {"anchor_count": 2, "descriptor_dim": 2}
    duplicate = _map()
    duplicate["anchor_ids"] = torch.tensor([3, 3])
    with pytest.raises(ValueError, match="unique"):
        validate_compact_map(duplicate)
    prototype = _map()
    prototype["anchor_extra_prototype_features"] = torch.ones(1, 2)
    with pytest.raises(ValueError, match="forbidden"):
        validate_compact_map(prototype)


def test_p0_query_parity_ignores_only_timing() -> None:
    left = [{"image_name": "q", "pose_w2c": [[1.0]], "frontend_ms": 1.0}]
    right = [{"image_name": "q", "pose_w2c": [[1.0]], "frontend_ms": 9.0}]
    assert compare_query_results(left, right)["non_timing_mismatch_count"] == 0
    right[0]["pose_w2c"] = [[2.0]]
    with pytest.raises(ValueError, match="pose_w2c"):
        compare_query_results(left, right)


def test_tensor_tree_parity_is_bitwise() -> None:
    assert tensor_tree_equal(_map(), _map())
    changed = _map()
    changed["anchor_xyz"][0, 0] = torch.nextafter(
        changed["anchor_xyz"][0, 0], torch.tensor(1.0)
    )
    assert not tensor_tree_equal(_map(), changed)


def test_formal_import_audit_rejects_non_allowlisted_and_forbidden(tmp_path: Path) -> None:
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "common").mkdir()
    entry = root / "scripts" / "main.py"
    entry.write_text("from common.safe import value\n")
    dependency = root / "common" / "safe.py"
    dependency.write_text("value = 1\n")
    allow = root / "allow.json"
    allow.write_text(json.dumps({"allowed_source_files": ["scripts/main.py"]}))
    with pytest.raises(ValueError, match="not allowlisted"):
        audit_formal_import_graph(root=root, entrypoint=entry, allowlist_path=allow)
    dependency.write_text("import map_learning.v6_proposals\n")
    allow.write_text(json.dumps({"allowed_source_files": ["scripts/main.py", "common/safe.py"]}))
    with pytest.raises(ValueError, match="forbidden"):
        audit_formal_import_graph(root=root, entrypoint=entry, allowlist_path=allow)
