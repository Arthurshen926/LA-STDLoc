import json
from types import SimpleNamespace

from train_lafgs_map import _cache_signature


def _dataset(model_path, source_path):
    return SimpleNamespace(
        model_path=str(model_path),
        source_path=str(source_path),
        feature_type="sp",
        images="processed",
        resolution=1,
        longest_edge=0,
        white_background=False,
        norm_before_render=True,
    )


def _args():
    return SimpleNamespace(
        observation_source="native",
        query_feature_contract="native_resized_input",
        load_iteration=30000,
        native_keypoint_count=2048,
    )


def test_query_cache_signature_changes_with_rgb_prior_identity(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    manifest_path = model_path / "rgb_prior_manifest.json"
    manifest = {
        "gaussian_type": "3dgs",
        "primitive_count": 100,
        "geometry_sha256": "geometry-a",
        "appearance_sha256": "appearance-a",
        "exported_ply_sha256": "ply-a",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    signature_a, payload_a = _cache_signature(
        _dataset(model_path, tmp_path / "scene"), _args()
    )
    manifest["geometry_sha256"] = "geometry-b"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    signature_b, payload_b = _cache_signature(
        _dataset(model_path, tmp_path / "scene"), _args()
    )

    assert signature_a != signature_b
    assert payload_a["version"] == 10
    assert payload_a["rgb_prior_fingerprint"]["geometry_sha256"] == "geometry-a"
    assert payload_b["rgb_prior_fingerprint"]["geometry_sha256"] == "geometry-b"
