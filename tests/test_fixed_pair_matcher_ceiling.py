from __future__ import annotations

from collections import OrderedDict
import copy
import json
from pathlib import Path

import pytest
import torch

from common.hashing import sha256_file
from evidence.fixed_pair_matcher_ceiling import (
    direct_lighterglue_match,
    materialize_paired_probe,
    pair_gate_report,
    validate_paired_probe,
)
from map_learning import fixed_pair_matcher_ceiling as feature_core
from map_learning.fixed_pair_matcher_ceiling import (
    BundledXFeatSpec,
    LIGHTGLUE_CONFIG,
    _load_module,
    _query_hashes,
    extract_bundled_features,
    feature_cache_content_sha256,
    load_bundled_checkpoint,
    pair_table_sha256,
    preregistration,
    resample_dense_teacher,
    strict_load_bundled_state,
    validate_bundled_xfeat_artifact,
    validate_feature_cache,
)
from scripts import aggregate_fixed_pair_matcher_ceiling_cross_scene as cross_cli
from scripts import compare_fixed_pair_matcher_ceiling as compare_cli
from scripts import materialize_fixed_pair_feature_cache as feature_cli
from scripts.fixed_pair_matcher_ceiling_common import (
    atomic_torch_save_fresh,
    load_completion,
    torch_load,
)
from features.multiview_fusion import sample_mask_at_grid_uv


@pytest.fixture(scope="module")
def bundled_artifact() -> dict:
    contract = preregistration()
    external = contract["external_xfeat"]
    checkpoint = contract["checkpoint"]
    spec = BundledXFeatSpec(
        worktree=Path(external["worktree"]),
        checkpoint=Path(checkpoint["path"]),
        expected_checkpoint_sha256=checkpoint["sha256"],
        expected_parent_commit=external["parent_commit"],
        expected_xfeat_tree=external["xfeat_tree"],
    )
    return validate_bundled_xfeat_artifact(spec)


def _fresh_models(artifact: dict):
    from kornia.feature.lightglue import LightGlue

    module = _load_module(Path(artifact["files"]["model"]["path"]), label="test")
    return module.XFeatModel().cpu().eval(), LightGlue(
        None, **LIGHTGLUE_CONFIG
    ).cpu().eval()


def test_preregistration_and_reviewed_implementation_registry_are_exact():
    payload = preregistration()
    assert payload["valid"] is True
    assert payload["checkpoint"]["exact_tensor_key_count"] == 291
    assert payload["checkpoint"]["exact_split"]["extractor"]["tensor_key_count"] == 122
    assert payload["checkpoint"]["exact_split"]["matcher"]["tensor_key_count"] == 169
    registry = feature_core.implementation_registry()
    assert registry["valid"] is True
    assert registry["implementation_commit"] == (
        "b63fc5e4e1306c150e59f029ce87780e8b0d6827"
    )
    assert registry["authorizes_real_mapping_pair_gate"] is True
    assert registry["authorizes_track"] is False
    assert registry["authorizes_test"] is False


def test_strict_bundled_loader_accepts_exact_291_key_split(bundled_artifact: dict):
    state = load_bundled_checkpoint(bundled_artifact["checkpoint"]["path"])
    extractor, matcher = _fresh_models(bundled_artifact)
    result = strict_load_bundled_state(state, extractor=extractor, matcher=matcher)
    assert result["extractor_key_count"] == 122
    assert result["matcher_checkpoint_key_count"] == 169
    assert result["matcher_runtime_key_count"] == 170
    assert result["strict_false_used"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_strict_bundled_loader_rejects_missing_or_extra_keys(
    bundled_artifact: dict, mutation: str
):
    state = load_bundled_checkpoint(bundled_artifact["checkpoint"]["path"])
    if mutation == "missing":
        state.pop(next(iter(state)))
    else:
        state["unexpected.tensor"] = torch.zeros(1)
    extractor, matcher = _fresh_models(bundled_artifact)
    with pytest.raises(ValueError, match="exact E1/LG axes"):
        strict_load_bundled_state(state, extractor=extractor, matcher=matcher)


def test_strict_bundled_loader_rejects_bn_or_axis_shape_mutation(
    bundled_artifact: dict,
):
    state = load_bundled_checkpoint(bundled_artifact["checkpoint"]["path"])
    key = "extractor.model.net.block1.0.layer.0.weight"
    state[key] = state[key].permute(1, 0, 2, 3).contiguous()
    extractor, matcher = _fresh_models(bundled_artifact)
    with pytest.raises(ValueError, match="shape/BN axis mismatch"):
        strict_load_bundled_state(state, extractor=extractor, matcher=matcher)


def test_strict_bundled_loader_rejects_bn_buffer_shape_mutation(
    bundled_artifact: dict,
):
    state = load_bundled_checkpoint(bundled_artifact["checkpoint"]["path"])
    key = "extractor.model.net.block1.0.layer.1.running_mean"
    state[key] = state[key][:-1].contiguous()
    extractor, matcher = _fresh_models(bundled_artifact)
    with pytest.raises(ValueError, match="shape/BN axis mismatch"):
        strict_load_bundled_state(state, extractor=extractor, matcher=matcher)


def test_strict_bundled_loader_rejects_runtime_confidence_buffer_mutation(
    bundled_artifact: dict,
):
    state = load_bundled_checkpoint(bundled_artifact["checkpoint"]["path"])
    extractor, matcher = _fresh_models(bundled_artifact)
    matcher.confidence_thresholds[0] += 0.01
    with pytest.raises(ValueError, match="confidence_thresholds"):
        strict_load_bundled_state(state, extractor=extractor, matcher=matcher)


def test_artifact_rejects_relative_xfeat_pt_checkpoint(bundled_artifact: dict):
    contract = preregistration()
    worktree = Path(contract["external_xfeat"]["worktree"])
    wrong = worktree / "encoders/XFeat/weights/xfeat.pt"
    spec = BundledXFeatSpec(
        worktree=worktree,
        checkpoint=wrong,
        expected_checkpoint_sha256=sha256_file(wrong),
        expected_parent_commit=contract["external_xfeat"]["parent_commit"],
        expected_xfeat_tree=contract["external_xfeat"]["xfeat_tree"],
    )
    with pytest.raises(ValueError, match="bundled xfeat-lighterglue"):
        validate_bundled_xfeat_artifact(spec)


class _SyntheticExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def forward(self, value):
        self.forward_count += 1
        height, width = value.shape[-2] // 8, value.shape[-1] // 8
        dense = torch.ones((1, 64, height, width), dtype=torch.float32)
        logits = torch.full((1, 65, height, width), -20.0)
        logits[:, 0] = 20.0
        reliability = torch.ones((1, 1, height, width), dtype=torch.float32)
        return dense, logits, reliability


class _NearestInterpolator(torch.nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def forward(self, value, positions, H, W):
        grid = 2.0 * positions / positions.new_tensor([W - 1, H - 1]) - 1.0
        sampled = torch.nn.functional.grid_sample(
            value,
            grid.unsqueeze(-2),
            mode=self.mode,
            align_corners=False,
        )
        return sampled.permute(0, 2, 3, 1).squeeze(-2)


def test_greatcourt_non_divisible_resize_preserves_raw_native_coordinates_and_depth():
    extractor = _SyntheticExtractor()
    interpolators = {
        mode: _NearestInterpolator(mode) for mode in ("nearest", "bilinear", "bicubic")
    }
    result = extract_bundled_features(
        extractor=extractor,
        interpolators=interpolators,
        native_image=torch.ones((3, 33, 65)),
        valid_mask=torch.ones((33, 65), dtype=torch.bool),
        requested_keypoint_count=16,
    )
    assert extractor.forward_count == 1
    assert result["detector_lineage"]["xfeat_input_hw"] == [32, 64]
    assert result["raw_xfeat_xy"].shape == result["native_xy"].shape
    scale = torch.tensor([65 / 64, 33 / 32])
    assert torch.allclose(result["native_xy"], result["raw_xfeat_xy"] * scale)
    assert result["descriptor"].shape[1] == 64

    depth, alpha, lineage = resample_dense_teacher(
        {
            "native_depth": torch.ones((16, 32)),
            "native_alpha": torch.ones((17, 33)),
        },
        (33, 65),
    )
    assert depth.shape == (33, 65)
    assert alpha.shape == (33, 65)
    assert lineage["source_native_depth_hw"] == [16, 32]
    assert lineage["target_native_hw"] == [33, 65]


def test_fractional_native_mask_lookup_uses_round_to_even_not_floor():
    mask = torch.zeros((3, 3), dtype=torch.bool)
    mask[0, 0] = True
    mask[2, 2] = True
    keep = sample_mask_at_grid_uv(
        mask,
        torch.tensor([[0.5, 0.5], [1.5, 1.5], [1.49, 1.49]]),
    )
    assert keep.tolist() == [True, True, False]


def _synthetic_feature_cache() -> dict:
    height, width = 64, 64
    intrinsics = torch.tensor([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    names = ["mapping-0.png", "mapping-1.png", "mapping-2.png"]
    centers = (0.0, 1.0, -1.0)
    world = torch.tensor([[0.0, 0.0, 5.0], [0.5, 0.5, 5.0]])
    queries = OrderedDict()
    for index, (name, center) in enumerate(zip(names, centers)):
        pose = torch.eye(4)
        pose[0, 3] = -center
        camera = world.clone()
        camera[:, 0] -= center
        physical = camera @ intrinsics.T
        native = physical[:, :2] / physical[:, 2, None] - 0.5
        descriptor = torch.zeros((2, 64))
        descriptor[0, 0] = 1.0
        descriptor[1, 1] = 1.0
        record = {
            "query_index": index,
            "query_name": name,
            "native_input_hw": [height, width],
            "xfeat_input_hw": [height, width],
            "row_count": 2,
            "raw_xfeat_xy": native.clone(),
            "native_xy": native,
            "descriptor": descriptor,
            "detector_score": torch.tensor([0.9, 0.8]),
            "native_depth_resampled": torch.full((height, width), 5.0),
            "native_alpha_resampled": torch.ones((height, width), dtype=torch.float16),
            "native_valid_mask": torch.ones((height, width), dtype=torch.bool),
            "native_K": intrinsics.clone(),
            "pose_w2c": pose,
            "image_lineage": {},
            "detector_lineage": {},
            "dense_teacher_lineage": {},
        }
        queries[name] = record
    names_sha = feature_core.ordered_names_sha256(names)
    contract = preregistration()
    scene = contract["fixed_scene_registry"]["stairs"]
    source = scene["mapping_source_image_manifest"]
    producer_identity = {"compiled_identity": "a" * 64}
    manifest_path = str(
        (Path(feature_core.__file__).resolve().parents[1] / source["path"]).resolve()
    )
    for record in queries.values():
        record["image_lineage"] = {
            "source_image_sha256": "1" * 64,
            "native_masked_rgb_sha256": "2" * 64,
            "native_input_hw": [height, width],
            "native_valid_mask_sha256": feature_core.tensor_sha256(
                record["native_valid_mask"]
            ),
        }
        record["detector_lineage"] = {
            "native_input_hw": [height, width],
            "xfeat_input_hw": [height, width],
            "single_model_forward": True,
            "detection_threshold_strict_greater_than": 0.05,
            "nms_kernel_size": 5,
            "sort": "descending_score_stable_row_major_ties",
            "mask_refill": False,
            "requested_top_k_before_mask": 2,
            "post_mask_count": 2,
            "raw_coordinates_preserved": True,
            "native_coordinates_scaled": True,
        }
        record["dense_teacher_lineage"] = {
            "target_native_hw": [height, width],
            "depth_resample": "identity_or_bilinear_align_corners_false",
            "alpha_resample": "identity_or_bilinear_align_corners_false",
            "alpha_threshold": 0.2,
        }
        record["hashes"] = _query_hashes(record)
    payload = {
        "schema": feature_core.FEATURE_CACHE_SCHEMA,
        "version": 1,
        "scene": "stairs",
        "mapping_only": True,
        "uses_test_queries": False,
        "run_uuid": "0" * 32,
        "query_count": len(names),
        "query_names": names,
        "query_names_sha256": names_sha,
        "requested_keypoint_count": 2,
        "inputs": {
            "query_cache": {
                "path": "/synthetic/query_cache.pt",
                "sha256": scene["query_cache"]["sha256"],
                "mapping_scope": {"uses_test_queries": False},
            },
            "mapping_source_images": {
                "manifest_path": manifest_path,
                "manifest_sha256": source["sha256"],
                "ordered_source_image_manifest_sha256": source[
                    "ordered_source_image_manifest_sha256"
                ],
                "mapping_image_count": source["mapping_image_count"],
                "mapping_image_total_bytes": source["mapping_image_total_bytes"],
                "mapping_test_name_intersection_count": 0,
            },
            "parent_stairs_gate": None,
        },
        "extractor": {
            "name": "E1_bundled_xfeat_extractor",
            "checkpoint": {
                "path": contract["checkpoint"]["path"],
                "sha256": contract["checkpoint"]["sha256"],
            },
            "external_parent_commit": contract["external_xfeat"]["parent_commit"],
            "xfeat_tree": contract["external_xfeat"]["xfeat_tree"],
            "model": {"sha256": contract["external_xfeat"]["files"]["model"]["sha256"]},
            "interpolator": {
                "sha256": contract["external_xfeat"]["files"]["interpolator"]["sha256"]
            },
            "state": {
                "checkpoint_key_count": 291,
                "extractor_key_count": 122,
                "matcher_checkpoint_key_count": 169,
                "matcher_runtime_key_count": 170,
                "runtime_only_buffer": "confidence_thresholds",
                "confidence_thresholds_sha256": (
                    feature_core.CONFIDENCE_THRESHOLDS_SHA256
                ),
                "extractor_strict_load": True,
                "matcher_strict_load": True,
                "strict_false_used": False,
            },
        },
        "feature_contract": {
            "single_forward_per_image": True,
            "detection_threshold": 0.05,
            "nms_kernel_size": 5,
            "nms_radius": 2,
            "top_k_before_mask": True,
            "mask_refill": False,
            "raw_and_native_coordinates_stored": True,
            "dense_depth_alpha_resampled": True,
            "greatcourt_non_divisible_by_32_supported": True,
        },
        "producer_identity": producer_identity,
        "producer": {
            "device": "cpu",
            "gpu_used": False,
            "query_count": 3,
            "model_forward_count": 3,
            "total_feature_rows": 6,
        },
        "queries": queries,
    }
    payload["content_sha256"] = feature_cache_content_sha256(payload)
    validate_feature_cache(payload)
    return payload


class _IdentityLightGlue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def forward(self, data):
        self.forward_count += 1
        count = min(
            data["image0"]["descriptors"].shape[1],
            data["image1"]["descriptors"].shape[1],
        )
        rows = torch.arange(count)
        return {
            "matches": [torch.stack([rows, rows], 1)],
            "scores": [torch.full((count,), 0.75)],
            "stop": 6,
        }


class _ForbiddenDetector:
    p9_forbidden_detector_sentinel = True

    def __call__(self):
        raise AssertionError("detector re-entry")


def _synthetic_probe() -> tuple[dict, dict, list[tuple[int, int]]]:
    cache = _synthetic_feature_cache()
    pairs = [(0, 1), (0, 2), (1, 2)]
    matcher = _IdentityLightGlue()
    probe = materialize_paired_probe(
        scene="stairs",
        feature_cache=cache,
        feature_cache_path="/synthetic/p9_fixed_pair_feature_cache.pt",
        feature_cache_sha256="c" * 64,
        pairs=pairs,
        proposal_lineage={
            "path": "/synthetic/proposals.pt",
            "sha256": "d" * 64,
            "content_sha256": "e" * 64,
            "arm": "nearest",
            "pair_table_sha256": pair_table_sha256(pairs),
            "match_rows_reused": False,
        },
        matcher=matcher,
        matcher_identity={"synthetic": True},
        run_uuid=cache["run_uuid"],
        producer_identity=cache["producer_identity"],
        detector_sentinel=_ForbiddenDetector(),
    )
    assert matcher.forward_count == len(pairs)
    return probe, cache, pairs


def test_pair_stage_never_calls_detector_and_uses_same_rows_for_both_arms():
    probe, _, _ = _synthetic_probe()
    assert probe["shared_contract"]["detector_forward_count_during_pair_stage"] == 0
    assert probe["feature_cache"]["same_exact_rows_for_both_arms"] is True
    for arm in probe["arms"].values():
        assert arm["metrics"]["correct_correspondence_count"] == 6
        assert arm["metrics"]["correct_correspondence_precision"] == 1.0
        assert arm["metrics"]["verified_keypoint_triangle_count"] == 2
        assert arm["metrics"]["cycle_supported_edge_count"] == 6
        assert arm["metrics"]["identity_conflict_count"] == 0


def test_feature_row_mutation_is_rejected():
    cache = _synthetic_feature_cache()
    cache["queries"]["mapping-0.png"]["descriptor"][0, 0] = 0.5
    with pytest.raises(ValueError, match="hash is stale|feature rows escape"):
        validate_feature_cache(cache)


@pytest.mark.parametrize("field", ["detector_score", "native_depth_resampled"])
def test_feature_cache_rejects_resigned_noncanonical_shapes(field: str):
    cache = _synthetic_feature_cache()
    query = cache["queries"]["mapping-0.png"]
    query[field] = query[field].unsqueeze(0)
    query["tensor_sha256"] = _query_hashes(query)
    cache["content_sha256"] = feature_cache_content_sha256(cache)
    with pytest.raises(ValueError, match="shape|misaligned"):
        validate_feature_cache(cache)


def test_probe_pair_or_match_row_mutation_is_rejected():
    probe, cache, pairs = _synthetic_probe()
    mutated = copy.deepcopy(probe)
    mutated["pair_table"]["right_query_index"][0] = 2
    with pytest.raises(ValueError, match="fixed pair table"):
        validate_paired_probe(mutated, feature_cache=cache, expected_pairs=pairs)
    mutated = copy.deepcopy(probe)
    mutated["arms"]["mnn_control"]["matches"]["source_row"][0] = 1
    with pytest.raises(ValueError, match="hash is stale|non-reciprocal"):
        validate_paired_probe(mutated, feature_cache=cache, expected_pairs=pairs)


@pytest.mark.parametrize("field", ["pair", "match", "diagnostic"])
def test_probe_rejects_noncanonical_column_shapes(field: str):
    probe, cache, pairs = _synthetic_probe()
    if field == "pair":
        probe["pair_table"]["left_query_index"] = probe["pair_table"][
            "left_query_index"
        ].unsqueeze(1)
    elif field == "match":
        probe["arms"]["mnn_control"]["matches"]["source_row"] = probe["arms"][
            "mnn_control"
        ]["matches"]["source_row"].unsqueeze(1)
    else:
        probe["arms"]["mnn_control"]["pair_diagnostics"]["raw_match_count"] = probe[
            "arms"
        ]["mnn_control"]["pair_diagnostics"]["raw_match_count"].unsqueeze(1)
    with pytest.raises(ValueError, match="exact shape|partial"):
        validate_paired_probe(probe, feature_cache=cache, expected_pairs=pairs)


@pytest.mark.parametrize("mutation", ["partial", "splice"])
def test_probe_rejects_partial_or_spliced_arms(mutation: str):
    probe, cache, pairs = _synthetic_probe()
    if mutation == "partial":
        del probe["arms"]["lighterglue_variant"]
    else:
        probe["arms"]["lighterglue_variant"]["run_uuid"] = "other-run"
    with pytest.raises(ValueError, match="schema/scope|spliced or are partial"):
        validate_paired_probe(probe, feature_cache=cache, expected_pairs=pairs)


def test_direct_lighterglue_is_one_forward_and_uses_no_images():
    cache = _synthetic_feature_cache()
    matcher = _IdentityLightGlue()
    left, right = list(cache["queries"].values())[:2]
    source, target, score, diagnostic = direct_lighterglue_match(matcher, left, right)
    assert matcher.forward_count == 1
    assert diagnostic["direct_forward_count"] == 1
    assert source.tolist() == target.tolist() == [0, 1]
    assert torch.all(score == 0.75)


def test_pair_gate_scientific_stop_and_no_downstream_authority():
    probe, _, _ = _synthetic_probe()
    producer = {"compiled_identity": "a" * 64}
    report = pair_gate_report(
        probe=probe,
        probe_path="/synthetic/fixed_pair_match_probe.pt",
        probe_sha256="f" * 64,
        completion_path="/synthetic/paired_match_completion.json",
        completion_sha256="1" * 64,
        producer_identity=producer,
        compiled_identity=producer["compiled_identity"],
        parent_stairs_gate=None,
    )
    assert report["scene_pair_gate_passed"] is False
    assert report["decision"] == "STOP_FIXED_PAIR_MATCHER_CEILING"
    assert report["gates"]["at_least_one_primary_strict_gain"] is False
    assert report["advance_to_track_implementation_review"] is False
    assert report["authorizes_real_track_run"] is False
    assert report["authorizes_test"] is False


def test_pair_gate_go_still_authorizes_review_only():
    probe, _, _ = _synthetic_probe()
    variant = probe["arms"]["lighterglue_variant"]["metrics"]
    variant["verified_keypoint_triangle_count"] += 1
    producer = {"compiled_identity": "a" * 64}
    report = pair_gate_report(
        probe=probe,
        probe_path="/synthetic/fixed_pair_match_probe.pt",
        probe_sha256="f" * 64,
        completion_path="/synthetic/paired_match_completion.json",
        completion_sha256="1" * 64,
        producer_identity=producer,
        compiled_identity=producer["compiled_identity"],
        parent_stairs_gate=None,
    )
    assert report["scene_pair_gate_passed"] is True
    assert report["decision"] == "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
    assert report["requires_other_scene"] is True
    assert report["advance_to_track_implementation_review"] is False


def test_pair_gate_uses_counts_and_rejects_edited_float_projection():
    probe, _, _ = _synthetic_probe()
    variant = probe["arms"]["lighterglue_variant"]["metrics"]
    variant["raw_match_count"] = 7
    variant["epipolar_accepted_count"] = 6
    variant["epipolar_acceptance_rate"] = 6 / 7
    variant["verified_keypoint_triangle_count"] += 1
    producer = {"compiled_identity": "a" * 64}
    report = pair_gate_report(
        probe=probe,
        probe_path="/synthetic/fixed_pair_match_probe.pt",
        probe_sha256="f" * 64,
        completion_path="/synthetic/paired_match_completion.json",
        completion_sha256="1" * 64,
        producer_identity=producer,
        compiled_identity=producer["compiled_identity"],
        parent_stairs_gate=None,
    )
    assert report["gates"]["epipolar_acceptance_rate_not_lower"] is False
    variant["epipolar_acceptance_rate"] = 1.0
    with pytest.raises(ValueError, match="differs from authoritative counts"):
        pair_gate_report(
            probe=probe,
            probe_path="/synthetic/fixed_pair_match_probe.pt",
            probe_sha256="f" * 64,
            completion_path="/synthetic/paired_match_completion.json",
            completion_sha256="1" * 64,
            producer_identity=producer,
            compiled_identity=producer["compiled_identity"],
            parent_stairs_gate=None,
        )


def test_atomic_fresh_file_rejects_partial_replacement(tmp_path: Path):
    output = tmp_path / "cache.pt"
    atomic_torch_save_fresh({"complete": True}, output)
    assert torch_load(output) == {"complete": True}
    with pytest.raises(FileExistsError, match="appeared|fresh"):
        atomic_torch_save_fresh({"complete": False}, output)
    assert torch_load(output) == {"complete": True}


def test_atomic_validator_failure_never_exposes_partial_output(tmp_path: Path):
    output = tmp_path / "cache.pt"

    def reject(_):
        raise ValueError("synthetic validation failure")

    with pytest.raises(ValueError, match="validation failure"):
        atomic_torch_save_fresh({"complete": False}, output, validator=reject)
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_partial_completion_is_rejected_before_artifact_loading(tmp_path: Path):
    path = tmp_path / "paired_match_completion.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lafgs_p9_fixed_pair_match_probe_completion",
                "version": 1,
                "scene": "stairs",
                "mapping_only": True,
                "uses_test_queries": False,
                "complete": False,
                "partial": True,
                "resume_allowed": False,
                "build_order": ["mnn_control"],
            }
        )
    )
    with pytest.raises(ValueError, match="partial or invalid"):
        load_completion(path=path, expected_file_sha256=sha256_file(path))


def test_completion_missing_failure_recovery_is_rejected_before_loading(
    tmp_path: Path,
):
    path = tmp_path / "paired_match_completion.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lafgs_p9_fixed_pair_match_probe_completion",
                "version": 1,
                "scene": "stairs",
                "mapping_only": True,
                "uses_test_queries": False,
                "complete": True,
                "partial": False,
                "resume_allowed": False,
                "run_uuid": "synthetic",
                "producer_identity": {},
                "compiled_identity": "a" * 64,
                "build_order": ["mnn_control", "lighterglue_variant"],
                "inputs": {},
                "artifacts": {},
            }
        )
    )
    with pytest.raises(ValueError, match="partial or invalid"):
        load_completion(path=path, expected_file_sha256=sha256_file(path))


def test_greatcourt_rejects_parent_identity_before_dataset_or_model_io(
    monkeypatch,
):
    monkeypatch.setattr(feature_cli, "configure_formal_cpu_runtime", lambda: None)
    monkeypatch.setattr(
        feature_cli,
        "producer_identity",
        lambda **_: {"compiled_identity": "a" * 64},
    )
    monkeypatch.setattr(
        feature_cli,
        "load_scene_gate",
        lambda **_: {
            "path": Path("/synthetic/stairs.json"),
            "sha256": "1" * 64,
            "scientific_projection": {
                "compiled_identity": "b" * 64,
            },
        },
    )
    monkeypatch.setattr(
        feature_cli,
        "require_fixed_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dataset I/O happened before parent identity rejection")
        ),
    )
    arguments = feature_cli.build_parser().parse_args(
        [
            "--scene",
            "greatcourt",
            "--dataset",
            "/synthetic/dataset",
            "--query-cache",
            "/synthetic/cache.pt",
            "--expected-query-cache-sha256",
            "1" * 64,
            "--mapping-scope-equivalence",
            "/synthetic/proof.json",
            "--expected-mapping-scope-equivalence-sha256",
            "2" * 64,
            "--xfeat-worktree",
            "/synthetic/xfeat",
            "--checkpoint",
            "/synthetic/checkpoint.pt",
            "--expected-checkpoint-sha256",
            "3" * 64,
            "--expected-parent-commit",
            "4" * 40,
            "--expected-xfeat-tree",
            "5" * 40,
            "--stairs-pair-gate",
            "/synthetic/stairs.json",
            "--expected-stairs-pair-gate-sha256",
            "6" * 64,
            "--output",
            "/synthetic/output.pt",
        ]
    )
    with pytest.raises(ValueError, match="different implementation"):
        feature_cli.run(arguments)


def test_comparator_exit_codes_are_two_for_scientific_stop_and_one_for_input_error(
    monkeypatch, tmp_path: Path
):
    arguments = [
        "--completion",
        str(tmp_path / "completion.json"),
        "--expected-completion-sha256",
        "a" * 64,
        "--output",
        str(tmp_path / "gate.json"),
    ]
    monkeypatch.setattr(
        compare_cli,
        "run",
        lambda args: {"scene_pair_gate_passed": False},
    )
    with pytest.raises(SystemExit) as scientific:
        compare_cli.entrypoint(arguments)
    assert scientific.value.code == 2

    def invalid(_):
        raise ValueError("bad lineage")

    monkeypatch.setattr(compare_cli, "run", invalid)
    with pytest.raises(SystemExit) as input_error:
        compare_cli.entrypoint(arguments)
    assert input_error.value.code == 1
    assert not (tmp_path / "gate.json").exists()


def _synthetic_scene_gate(scene: str, *, passed: bool) -> dict:
    identity = {
        "compiled_identity": "a" * 64,
        "source_file_sha256": {"source.py": "b" * 64},
        "runtime": {"device": "cpu"},
    }
    path = Path(f"/synthetic/{scene}/p9_fixed_pair_matcher_ceiling_pair_gate.json")
    wrapped = {
        "path": path,
        "sha256": ("1" if scene == "stairs" else "2") * 64,
        "scientific_projection": {
            "scene": scene,
            "scene_pair_gate_passed": passed,
            "decision": (
                "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
                if passed
                else "STOP_FIXED_PAIR_MATCHER_CEILING"
            ),
            "compiled_identity": "a" * 64,
            "fixed_pair_table_sha256": "3" * 64,
        },
    }
    wrapped["payload"] = {
        "scene_pair_gate_passed": passed,
        "producer_identity": identity,
        "parent_stairs_gate": None,
    }
    return wrapped


def test_cross_scene_gate_is_review_only_and_parent_bound():
    stairs = _synthetic_scene_gate("stairs", passed=True)
    greatcourt = _synthetic_scene_gate("greatcourt", passed=True)
    greatcourt["payload"]["parent_stairs_gate"] = cross_cli._gate_input(stairs)
    report = cross_cli.cross_scene_report(
        stairs=stairs,
        greatcourt=greatcourt,
        producer=stairs["payload"]["producer_identity"],
    )
    assert report["both_scene_pair_gate_passed"] is True
    assert report["advance_to_track_implementation_review"] is True
    assert report["authorizes_real_track_run"] is False
    assert report["advance_to_pose"] is False
    assert report["authorizes_test"] is False

    greatcourt["payload"]["parent_stairs_gate"]["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="not bound"):
        cross_cli.cross_scene_report(
            stairs=stairs,
            greatcourt=greatcourt,
            producer=stairs["payload"]["producer_identity"],
        )


def test_cross_scene_exit_codes_are_two_for_stop_and_one_for_invalid(
    monkeypatch,
):
    arguments = [
        "--stairs-pair-gate",
        "/synthetic/stairs.json",
        "--expected-stairs-pair-gate-sha256",
        "a" * 64,
        "--greatcourt-pair-gate",
        "/synthetic/greatcourt.json",
        "--expected-greatcourt-pair-gate-sha256",
        "b" * 64,
        "--output",
        "/synthetic/cross.json",
    ]
    monkeypatch.setattr(
        cross_cli,
        "run",
        lambda args: {"both_scene_pair_gate_passed": False},
    )
    with pytest.raises(SystemExit) as scientific:
        cross_cli.entrypoint(arguments)
    assert scientific.value.code == 2

    def invalid(_):
        raise ValueError("bad lineage")

    monkeypatch.setattr(cross_cli, "run", invalid)
    with pytest.raises(SystemExit) as input_error:
        cross_cli.entrypoint(arguments)
    assert input_error.value.code == 1
