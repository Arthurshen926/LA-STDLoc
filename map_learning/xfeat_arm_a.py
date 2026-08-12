"""CPU-only, mapping-only XFeat Arm-A detector materialization.

This module reuses the audited Arm-B artifact, reference-registry, and native
RGB reconstruction.  It executes only XFeat's single-image sparse detector;
descriptors and pair matchers are deliberately outside this producer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from features.multiview_fusion import sample_mask_at_grid_uv
from map_learning.frontend_upper_bound import (
    PROBE_SCHEMA,
    PROBE_VERSION,
    file_sha256,
    tensor_sha256,
    validate_probe,
)
from map_learning.xfeat_arm_b import (
    COORDINATE_CONVENTION,
    XFeatArtifactSpec,
    _load_module,
    _query_names_sha256,
    load_xfeat_cpu,
    recreate_native_input,
    validate_reference_context,
    validate_xfeat_artifact,
    xfeat_resize_contract,
)


PRODUCER_SCHEMA = "lafgs_xfeat_arm_a_producer"
PRODUCER_VERSION = 1
DETECTION_THRESHOLD = 0.05
NMS_KERNEL_SIZE = 5
SCORE_SEMANTICS = "nearest_probability_times_bilinear_reliability"


def _producer_code_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    relatives = (
        "map_learning/xfeat_arm_a.py",
        "map_learning/xfeat_arm_b.py",
        "map_learning/frontend_upper_bound.py",
        "data/datasets.py",
        "data/images.py",
        "features/multiview_fusion.py",
    )
    return {
        relative: {
            "path": str((root / relative).resolve()),
            "sha256": file_sha256(root / relative),
        }
        for relative in relatives
    }


def _load_detector_interpolators(artifact: Mapping) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Instantiate the locked wrapper's nearest/bilinear samplers on CPU."""
    module = _load_module(
        Path(artifact["files"]["interpolator"]["path"]),
        identity="arm_a_interpolator",
    )
    interpolator_class = getattr(module, "InterpolateSparse2d", None)
    if interpolator_class is None:
        raise ValueError("locked XFeat implementation lacks InterpolateSparse2d")
    nearest = interpolator_class("nearest").cpu().eval()
    bilinear = interpolator_class("bilinear").cpu().eval()
    if str(getattr(nearest, "mode", "")) != "nearest":
        raise ValueError("XFeat detector probability sampler is not nearest")
    if str(getattr(bilinear, "mode", "")) != "bilinear":
        raise ValueError("XFeat detector reliability sampler is not bilinear")
    return nearest, bilinear


def xfeat_keypoint_heatmap(logits: torch.Tensor) -> torch.Tensor:
    """Apply the locked wrapper's 65-way softmax and 8x8 cell unpacking."""
    value = torch.as_tensor(logits).detach().cpu().float()
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 65:
        raise ValueError("XFeat keypoint logits must have shape [1,65,H/8,W/8]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("XFeat keypoint logits are non-finite")
    probabilities = F.softmax(value, dim=1)[:, :64]
    batch, _, height, width = probabilities.shape
    heatmap = probabilities.permute(0, 2, 3, 1).reshape(
        batch, height, width, 8, 8
    )
    return heatmap.permute(0, 1, 3, 2, 4).reshape(
        batch, 1, height * 8, width * 8
    )


def _mask_equivalence_proof(
    *,
    contract: Mapping,
    xfeat_keypoints: torch.Tensor,
    native_keypoints: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Prove bootstrap round() and consumer floor() mask lookups coincide.

    Arm A intentionally fails closed on non-identity XFeat resizing.  This is
    the preregistered safe case for Stairs (480x640): detector coordinates are
    integer native-grid coordinates, so round and floor indices and decisions
    must be exactly equal for every pre-mask row.
    """
    native_hw = tuple(int(value) for value in contract["native_input_hw"])
    xfeat_hw = tuple(int(value) for value in contract["xfeat_input_hw"])
    if native_hw != xfeat_hw or float(contract["rw"]) != 1.0 or float(contract["rh"]) != 1.0:
        raise ValueError(
            "Arm A mask contract requires native H/W divisible by 32 and identity XFeat resize"
        )
    if native_hw[0] % 32 or native_hw[1] % 32:
        raise ValueError("Arm A identity-resize dimensions are not divisible by 32")
    xfeat = torch.as_tensor(xfeat_keypoints).detach().cpu().float()
    native = torch.as_tensor(native_keypoints).detach().cpu().float()
    if tuple(xfeat.shape) != tuple(native.shape) or xfeat.ndim != 2 or xfeat.shape[1] != 2:
        raise ValueError("XFeat/native detector coordinate registries differ")
    if not torch.equal(xfeat, xfeat.round()) or not torch.equal(native, xfeat):
        raise ValueError("identity-resize XFeat coordinates are not exact native integers")
    round_indices = native.round().long()
    floor_indices = native.floor().long()
    if not torch.equal(round_indices, floor_indices):
        raise ValueError("round/floor candidate mask indices differ")
    height, width = native_hw
    inside = (
        (floor_indices[:, 0] >= 0)
        & (floor_indices[:, 0] < width)
        & (floor_indices[:, 1] >= 0)
        & (floor_indices[:, 1] < height)
    )
    if not bool(inside.all()):
        raise ValueError("candidate detector coordinate escapes the native image")
    mask = torch.as_tensor(valid_mask).detach().cpu().bool().squeeze()
    if tuple(mask.shape) != native_hw:
        raise ValueError("native detector mask shape differs from the resize contract")
    bootstrap_keep = sample_mask_at_grid_uv(mask, native)
    consumer_keep = mask[floor_indices[:, 1], floor_indices[:, 0]]
    if not torch.equal(bootstrap_keep, consumer_keep):
        raise ValueError("round/floor candidate mask decisions differ")
    return bootstrap_keep, {
        "required_native_hw_divisible_by_32": True,
        "native_hw_divisible_by_32": True,
        "identity_xfeat_resize": True,
        "integer_xfeat_coordinates": True,
        "round_floor_indices_equal": True,
        "round_floor_mask_decisions_equal": True,
        "checked_pre_mask_rows": int(native.shape[0]),
        "round_indices_sha256": tensor_sha256(round_indices),
        "floor_indices_sha256": tensor_sha256(floor_indices),
        "round_mask_keep_sha256": tensor_sha256(bootstrap_keep),
        "floor_mask_keep_sha256": tensor_sha256(consumer_keep),
    }


def extract_xfeat_detector(
    *,
    outputs: Sequence[torch.Tensor],
    native_hw: Sequence[int],
    valid_mask: torch.Tensor,
    requested_keypoint_count: int,
    nearest_interpolator: torch.nn.Module,
    bilinear_interpolator: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
    """Run the locked XFeat detector semantics on one model-forward result."""
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError("XFeat model must return descriptor/logit/reliability outputs")
    requested_k = int(requested_keypoint_count)
    if requested_k <= 0:
        raise ValueError("requested XFeat keypoint count must be positive")
    contract = xfeat_resize_contract(native_hw)
    xfeat_height, xfeat_width = (
        int(value) for value in contract["xfeat_input_hw"]
    )
    dense = torch.as_tensor(outputs[0]).detach().cpu()
    logits = torch.as_tensor(outputs[1]).detach().cpu().float()
    reliability = torch.as_tensor(outputs[2]).detach().cpu().float()
    expected_spatial = (xfeat_height // 8, xfeat_width // 8)
    if dense.ndim != 4 or dense.shape[0] != 1 or tuple(dense.shape[-2:]) != expected_spatial:
        raise ValueError("XFeat descriptor output has the wrong detector-grid shape")
    if tuple(logits.shape) != (1, 65, *expected_spatial):
        raise ValueError("XFeat logit output has the wrong detector-grid shape")
    if tuple(reliability.shape) != (1, 1, *expected_spatial):
        raise ValueError("XFeat reliability output has the wrong detector-grid shape")
    if not bool(torch.isfinite(reliability).all()):
        raise ValueError("XFeat reliability output is non-finite")

    heatmap = xfeat_keypoint_heatmap(logits)
    local_maximum = F.max_pool2d(
        heatmap,
        kernel_size=NMS_KERNEL_SIZE,
        stride=1,
        padding=NMS_KERNEL_SIZE // 2,
    )
    selected = (heatmap == local_maximum) & (heatmap > DETECTION_THRESHOLD)
    # nonzero is row-major; stable sorting below therefore makes exact score
    # ties deterministic while retaining the locked wrapper's ranking rule.
    yx = selected[0, 0].nonzero(as_tuple=False)
    xfeat_keypoints = yx[:, [1, 0]].float().contiguous()
    threshold_nms_count = int(xfeat_keypoints.shape[0])
    if threshold_nms_count:
        positions = xfeat_keypoints[None]
        probability = nearest_interpolator(
            heatmap, positions, H=xfeat_height, W=xfeat_width
        )[0].reshape(-1).float()
        reliability_score = bilinear_interpolator(
            reliability, positions, H=xfeat_height, W=xfeat_width
        )[0].reshape(-1).float()
        scores = probability * reliability_score
        sentinel = torch.all(xfeat_keypoints == 0, dim=1)
        scores[sentinel] = -1.0
        order = torch.argsort(-scores, stable=True)[:requested_k]
        xfeat_keypoints = xfeat_keypoints[order]
        scores = scores[order]
        positive = scores > 0
        xfeat_keypoints = xfeat_keypoints[positive]
        scores = scores[positive]
    else:
        probability = torch.empty(0, dtype=torch.float32)
        reliability_score = torch.empty(0, dtype=torch.float32)
        sentinel = torch.empty(0, dtype=torch.bool)
        scores = torch.empty(0, dtype=torch.float32)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("XFeat combined detector scores are non-finite")
    before_mask = int(xfeat_keypoints.shape[0])
    scale = xfeat_keypoints.new_tensor(
        [float(contract["rw"]), float(contract["rh"])]
    )
    native_keypoints = (xfeat_keypoints * scale).contiguous()
    keep, proof = _mask_equivalence_proof(
        contract=contract,
        xfeat_keypoints=xfeat_keypoints,
        native_keypoints=native_keypoints,
        valid_mask=valid_mask,
    )
    native_keypoints = native_keypoints[keep].contiguous()
    scores = scores[keep].contiguous()
    if scores.numel() > 1 and not bool((scores[:-1] >= scores[1:]).all()):
        raise RuntimeError("XFeat detector scores lost top-K order after masking")
    return native_keypoints, scores, before_mask, {
        **contract,
        "keypoint_heatmap": "softmax_65_discard_dustbin_then_8x8_unpack",
        "nms_kernel_size": NMS_KERNEL_SIZE,
        "nms_radius": NMS_KERNEL_SIZE // 2,
        "nms_passes": 1,
        "strict_probability_threshold": DETECTION_THRESHOLD,
        "score_semantics": SCORE_SEMANTICS,
        "probability_interpolation": "locked_XFeat_nearest",
        "reliability_interpolation": "locked_XFeat_bilinear",
        "interpolator_align_corners": False,
        "interpolator_normgrid": "(2*x/(W-1)-1,2*y/(H-1)-1)",
        "origin_padding_sentinel_excluded": True,
        "sort": "descending_score_stable_row_major_ties",
        "top_k_before_native_mask": requested_k,
        "candidate_count_after_threshold_nms": threshold_nms_count,
        "origin_sentinel_count_after_threshold_nms": int(sentinel.sum()),
        "positive_top_k_count_before_mask": before_mask,
        "post_mask_count": int(native_keypoints.shape[0]),
        "native_mask_filter": "sample_mask_at_grid_uv_nearest_round",
        "mask_equivalence_proof": proof,
        "shared_forward_descriptor_output_used": False,
        "candidate_descriptor_rows_materialized": False,
        "pair_matcher_used": False,
    }


def materialize_xfeat_arm_a(
    *,
    query_cache: Mapping,
    teacher: Mapping,
    query_cache_path: str | Path,
    teacher_path: str | Path,
    dataset_root: str | Path,
    artifact_spec: XFeatArtifactSpec,
) -> dict:
    """Materialize one complete consumer-valid Arm-A probe in memory."""
    query_cache_path = Path(query_cache_path).expanduser().resolve()
    teacher_path = Path(teacher_path).expanduser().resolve()
    if not query_cache_path.is_file() or not teacher_path.is_file():
        raise FileNotFoundError("query cache and teacher must be local files")
    artifact = validate_xfeat_artifact(artifact_spec)
    context = validate_reference_context(
        query_cache, teacher, dataset_root=dataset_root
    )
    model, _, state_summary = load_xfeat_cpu(artifact)
    nearest, bilinear = _load_detector_interpolators(artifact)
    queries = {}
    total_before_mask = 0
    total_after_mask = 0
    for query_index, (name, camera) in enumerate(zip(context.names, context.cameras)):
        cached = context.queries[name]
        reference_keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).detach().cpu().float()
        native_image, valid_mask, image_lineage = recreate_native_input(
            context, camera, cached
        )
        resize = xfeat_resize_contract(native_image.shape[-2:])
        xfeat_hw = tuple(int(value) for value in resize["xfeat_input_hw"])
        model_input = F.interpolate(
            native_image[None].cpu().float(),
            size=xfeat_hw,
            mode="bilinear",
            align_corners=False,
        )
        with torch.inference_mode():
            outputs = model(model_input)
            detector_keypoints, detector_scores, before_mask, detector_lineage = (
                extract_xfeat_detector(
                    outputs=outputs,
                    native_hw=native_image.shape[-2:],
                    valid_mask=valid_mask,
                    requested_keypoint_count=context.requested_keypoint_count,
                    nearest_interpolator=nearest,
                    bilinear_interpolator=bilinear,
                )
            )
        queries[name] = {
            "query_index": int(query_index),
            "query_name": name,
            "reference_keypoints_sha256": tensor_sha256(reference_keypoints),
            "detector_keypoints": detector_keypoints,
            "detector_keypoints_sha256": tensor_sha256(detector_keypoints),
            "detector_scores": detector_scores,
            "detector_scores_sha256": tensor_sha256(detector_scores),
            "detected_count_before_mask": before_mask,
            "detected_count_after_mask": int(detector_keypoints.shape[0]),
            "image_lineage": image_lineage,
            "detector_lineage": detector_lineage,
        }
        total_before_mask += before_mask
        total_after_mask += int(detector_keypoints.shape[0])

    final_artifact = validate_xfeat_artifact(artifact_spec)
    if final_artifact != artifact:
        raise RuntimeError("XFeat artifact changed while materializing Arm A")
    weights = artifact["files"]["weights"]
    implementation_id = (
        f"xfeat_tree_{artifact['xfeat_tree'][:12]}"
        f"__model_{artifact['files']['model']['sha256'][:12]}"
        "__arm_a_v1"
    )
    probe = {
        "schema": PROBE_SCHEMA,
        "version": PROBE_VERSION,
        "mapping_only": True,
        "uses_test_queries": False,
        "reference": {
            "query_cache_path": str(query_cache_path),
            "query_cache_sha256": file_sha256(query_cache_path),
            "query_cache_signature": query_cache["signature"],
            "teacher_path": str(teacher_path),
            "teacher_sha256": file_sha256(teacher_path),
            "teacher_schema": teacher["schema"],
            "query_names": list(context.names),
            "query_names_sha256": _query_names_sha256(context.names),
            "reference_rows": "exact_cached_superpoint_registry_hash_only",
        },
        "frontend": {
            "name": "xfeat_sparse_64d_detector_only",
            "family": "independent_local_frontend",
            "implementation_id": implementation_id,
            "coordinate_convention": COORDINATE_CONVENTION,
            "descriptor_dim": 64,
            "requested_keypoint_count": context.requested_keypoint_count,
            "weights": {
                "path": weights["path"],
                "sha256": weights["sha256"],
                "size_bytes": weights["size_bytes"],
            },
            "code": {
                "external_worktree": artifact["worktree"],
                "external_parent_commit": artifact["parent_commit"],
                "xfeat_tree": artifact["xfeat_tree"],
                "git_clean": artifact["git_clean"],
                "model": artifact["files"]["model"],
                "interpolator": artifact["files"]["interpolator"],
                "wrapper": artifact["files"]["wrapper"],
            },
        },
        "capabilities": {
            "detector_repeatability": True,
            "descriptor_identity": False,
        },
        "producer": {
            "schema": PRODUCER_SCHEMA,
            "version": PRODUCER_VERSION,
            "arm": "A_detector_repeatability",
            "device": "cpu",
            "dtype": "float32",
            "gpu_used": False,
            "network_access_used": False,
            "candidate_detector_used": True,
            "shared_forward_descriptor_output_used": False,
            "candidate_descriptor_rows_materialized": False,
            "pair_matcher_used": False,
            "implementation_files": _producer_code_identity(),
            "state_dict": state_summary,
            "query_count": len(context.names),
            "detected_count_before_mask": total_before_mask,
            "detected_count_after_mask": total_after_mask,
            "all_queries_identity_xfeat_resize": True,
            "all_queries_round_floor_mask_equivalent": True,
        },
        "queries": queries,
    }
    validation = validate_probe(
        probe,
        query_cache,
        teacher,
        require_detector=True,
        verify_weight_artifact=True,
        query_cache_path=query_cache_path,
        teacher_path=teacher_path,
    )
    if validation["validated_detector_keypoints"] != total_after_mask:
        raise RuntimeError("consumer validation changed the detector row count")
    probe["producer"]["consumer_validation"] = validation
    return probe
