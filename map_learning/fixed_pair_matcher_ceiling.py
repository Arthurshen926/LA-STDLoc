"""Strict P9 bundled-XFeat loader and mapping-only feature-cache producer.

The paired matcher experiment consumes one immutable feature cache.  Feature
extraction and pair matching are deliberately separated so a pair job cannot
silently re-detect either image.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import subprocess
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F

from common.hashing import canonical_json, sha256_file
from data.datasets import ColmapDataset
from data.images import resolution_from_longest_edge
from features.multiview_fusion import sample_mask_at_grid_uv


FEATURE_CACHE_SCHEMA = "lafgs_p9_fixed_pair_feature_cache"
FEATURE_CACHE_VERSION = 1
PREREGISTRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/p9_fixed_pair_matcher_ceiling_preregistration.json"
)
PREREGISTRATION_COMMIT = "ee638ce009f6b76f6393c8e9867198c46a434f82"
PREREGISTRATION_BLOB_SHA256 = (
    "31e775a3b1c1d418ef1ad03a42b46c652f477f8ab38dadf53fcd85efbb96950a"
)
PREREGISTRATION_AMENDMENT_COMMIT = "35ea06953d6a080e8ce1e609a60522bae74069ed"
IMPLEMENTATION_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/p9_fixed_pair_matcher_ceiling_implementation.json"
)
RAW_KEY_ORDER_SHA256 = (
    "76542560a0d12c49b2bef38cc1041450759fd8a857cb3788b61784a036f8232e"
)
EXTRACTOR_KEY_ORDER_SHA256 = (
    "7a1831132282b855068949cf02ed0e6a394f21da0f29032476e41bb42ab5abc6"
)
MATCHER_KEY_ORDER_SHA256 = (
    "bebf26fe084d5ebd9b71f71edb6dbd0cde65275c4df88e5278758e6a3966345d"
)
CONFIDENCE_THRESHOLDS_SHA256 = (
    "5182f70851806dfdb0b2942073b6669e66e37276f311aaaaea4cde75b3b7d3e5"
)
LIGHTGLUE_CONFIG_SHA256 = (
    "b2268682d92af67e2c3b11e66c17e30dea6e4aeeb546df21e6bc390b8f381585"
)
LIGHTGLUE_SOURCE_SHA256 = (
    "bb003885e5c4918d5762c6c98f9c423d61dcefe62790b8900d4fbb9bc738a7cb"
)
EXTRACTOR_PREFIX = "extractor.model.net."
MATCHER_PREFIX = "matcher."
EXTRACTOR_KEY_COUNT = 122
MATCHER_KEY_COUNT = 169
CHECKPOINT_KEY_COUNT = 291
DESCRIPTOR_DIM = 64
DETECTION_THRESHOLD = 0.05
NMS_KERNEL_SIZE = 5
ALPHA_THRESHOLD = 0.2
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

LIGHTGLUE_CONFIG: dict[str, Any] = {
    "name": "xfeat",
    "input_dim": 64,
    "descriptor_dim": 96,
    "add_scale_ori": False,
    "add_laf": False,
    "scale_coef": 1.0,
    "n_layers": 6,
    "num_heads": 1,
    "flash": True,
    "mp": False,
    "depth_confidence": -1,
    "width_confidence": 0.95,
    "filter_threshold": 0.1,
    "weights": None,
}

PRODUCER_SOURCE_PATHS = (
    "common/hashing.py",
    "data/colmap.py",
    "data/datasets.py",
    "data/images.py",
    "evidence/fixed_pair_matcher_ceiling.py",
    "features/multiview_fusion.py",
    "map_learning/frontend_upper_bound.py",
    "map_learning/fixed_pair_matcher_ceiling.py",
    "scripts/fixed_pair_matcher_ceiling_common.py",
    "scripts/materialize_fixed_pair_feature_cache.py",
    "scripts/materialize_fixed_pair_match_probe.py",
    "scripts/compare_fixed_pair_matcher_ceiling.py",
    "scripts/aggregate_fixed_pair_matcher_ceiling_cross_scene.py",
    "docs/evidence/p9_fixed_pair_matcher_ceiling_preregistration.json",
    "docs/evidence/p9_stairs_mapping_image_manifest.json",
    "docs/evidence/p9_greatcourt_mapping_image_manifest.json",
)


@dataclass(frozen=True)
class BundledXFeatSpec:
    """Exact local external checkout and bundled checkpoint identity."""

    worktree: Path
    checkpoint: Path
    expected_checkpoint_sha256: str
    expected_parent_commit: str
    expected_xfeat_tree: str


@dataclass(frozen=True)
class MappingFeatureContext:
    """Validated mapping-only image/cache registry."""

    dataset: ColmapDataset
    cameras: tuple
    names: tuple[str, ...]
    records: Mapping[str, Mapping]
    signature_payload: Mapping
    mapping_scope: Mapping
    query_cache_path: Path
    query_cache_sha256: str
    query_names_sha256: str
    requested_keypoint_count: int
    source_image_manifest: Mapping


class _CPUOnlyUnpickler(pickle.Unpickler):
    """Remap legacy CUDA-backed scene-mask tensors to CPU while unpickling."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(
                io.BytesIO(value), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


class _CPUColmapDataset(ColmapDataset):
    """COLMAP dataset variant whose legacy mask pickle cannot activate CUDA."""

    def _load_masks(self):
        for path in (
            self.root / self.images / "masks.pkl",
            self.root / "masks.pkl",
        ):
            if path.is_file():
                with path.open("rb") as handle:
                    return _CPUOnlyUnpickler(handle).load()
        return None


def _key_order_sha256(keys: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(str(key) for key in keys) + "\n").encode("utf-8")
    ).hexdigest()


def _tensor_bytes_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash dtype, exact shape, and contiguous CPU bytes."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def ordered_names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(str(name) for name in names) + "\n").encode("utf-8")
    ).hexdigest()


def pair_table_sha256(pairs: Sequence[Sequence[int]]) -> str:
    canonical = [[int(pair[0]), int(pair[1])] for pair in pairs]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def preregistration() -> dict:
    """Load the frozen P9 contract and reject an edited preregistration."""
    if sha256_file(PREREGISTRATION_PATH) != PREREGISTRATION_BLOB_SHA256:
        raise RuntimeError("P9 preregistration blob differs from its frozen commit")
    payload = json.loads(PREREGISTRATION_PATH.read_text())
    checkpoint = payload.get("checkpoint", {})
    split = checkpoint.get("exact_split", {})
    feature = payload.get("feature_cache", {})
    pair_probe = payload.get("fixed_pair_probe", {})
    artifact_schemas = payload.get("artifact_schemas", {})
    two_scene = payload.get("two_scene_protocol", {})
    if (
        payload.get("schema") != "lafgs_p9_fixed_pair_matcher_ceiling_preregistration"
        or payload.get("version") != 1
        or payload.get("valid") is not True
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or checkpoint.get("exact_tensor_key_count") != CHECKPOINT_KEY_COUNT
        or checkpoint.get("raw_key_order_sha256") != RAW_KEY_ORDER_SHA256
        or split.get("extractor", {}).get("tensor_key_count") != EXTRACTOR_KEY_COUNT
        or split.get("extractor", {}).get("stripped_key_order_sha256")
        != EXTRACTOR_KEY_ORDER_SHA256
        or split.get("matcher", {}).get("tensor_key_count") != MATCHER_KEY_COUNT
        or split.get("matcher", {}).get("stripped_key_order_sha256")
        != MATCHER_KEY_ORDER_SHA256
        or payload.get("lighterglue", {}).get("canonical_config_sha256")
        != LIGHTGLUE_CONFIG_SHA256
        or payload.get("lighterglue", {})
        .get("unique_runtime_buffer", {})
        .get("tensor_bytes_sha256")
        != CONFIDENCE_THRESHOLDS_SHA256
        or feature.get("schema") != FEATURE_CACHE_SCHEMA
        or feature.get("greatcourt_non_divisible_by_32_supported") is not True
        or pair_probe.get("p8_probe_reused") is not False
        or pair_probe.get("detector_reentry_forbidden") is not True
        or pair_probe.get("lighterglue_input", {}).get("keypoints")
        != "native_xy_without_pixel_center_offset"
        or payload.get("pair_gate", {}).get("metric_pipeline_order")
        != [
            "raw_matcher_output",
            "common_symmetric_epipolar_filter",
            "dense_depth_teacher_correctness",
            "exact_triangle_and_cycle_deduplication",
            "nonempty_pair_coverage",
            "identity_conflict_union",
        ]
        or two_scene.get("scene_order") != ["stairs", "greatcourt"]
        or two_scene.get("cross_scene_gate", {}).get("pass_decision")
        != "GO_TO_FIXED_PAIR_TRACK_IMPLEMENTATION_REVIEW"
        or artifact_schemas.get("paired_completion", {}).get(
            "only_authoritative_pair_gate_input"
        )
        is not True
        or payload.get("planned_producer_source_paths") != list(PRODUCER_SOURCE_PATHS)
    ):
        raise RuntimeError("P9 preregistration is structurally invalid")
    return payload


def load_source_image_manifest(*, scene: str) -> dict:
    """Load and bind the preregistered per-file mapping RGB manifest."""
    contract = preregistration()["fixed_scene_registry"][str(scene)]
    expected = contract["mapping_source_image_manifest"]
    root = Path(__file__).resolve().parents[1]
    path = (root / expected["path"]).resolve()
    if not path.is_file() or sha256_file(path) != expected["sha256"]:
        raise ValueError(
            "P9 mapping source-image manifest differs from preregistration"
        )
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "lafgs_p9_mapping_source_image_manifest"
        or payload.get("version") != 1
        or payload.get("valid") is not True
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("scene") != scene
        or Path(str(payload.get("dataset_root", ""))).resolve()
        != Path(contract["dataset_root"]).resolve()
        or payload.get("images") != contract["images"]
        or payload.get("mapping_image_count") != expected["mapping_image_count"]
        or payload.get("mapping_image_total_bytes")
        != expected["mapping_image_total_bytes"]
        or payload.get("ordered_source_image_manifest_sha256")
        != expected["ordered_source_image_manifest_sha256"]
        or payload.get("ordered_source_names_sha256") != contract["query_names_sha256"]
        or payload.get("revalidation", {}).get("mapping_test_name_intersection_count")
        != 0
    ):
        raise ValueError("P9 mapping source-image manifest is structurally invalid")
    dataset_root = Path(payload["dataset_root"]).resolve()
    split_and_masks = payload.get("split_and_mask_files")
    if not isinstance(split_and_masks, Mapping) or not split_and_masks:
        raise ValueError("P9 mapping source-image manifest lacks split/mask lineage")
    for relative, identity in split_and_masks.items():
        candidate = (dataset_root / _safe_name(relative)).resolve()
        try:
            candidate.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError("P9 split/mask lineage escapes the dataset") from error
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(identity.get("size_bytes", -1))
            or sha256_file(candidate) != identity.get("sha256")
        ):
            raise ValueError(
                "P9 dataset split/mask lineage differs from preregistration"
            )
    return {"path": str(path), "sha256": expected["sha256"], **payload}


def validate_source_images(
    *, context: MappingFeatureContext, expected_manifest: Mapping
) -> dict:
    """Recompute the exact ordered mapping RGB digest and split boundary."""
    digest = hashlib.sha256()
    total_bytes = 0
    test_names = set(getattr(context.dataset, "_test_names", frozenset()))
    intersection = 0
    for name, camera in zip(context.names, context.cameras):
        path = Path(camera.image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"P9 source mapping image is missing: {path}")
        file_digest = sha256_file(path)
        digest.update(f"{name}\t{file_digest}\n".encode("utf-8"))
        total_bytes += int(path.stat().st_size)
        intersection += int(name in test_names)
    if (
        digest.hexdigest() != expected_manifest["ordered_source_image_manifest_sha256"]
        or total_bytes != int(expected_manifest["mapping_image_total_bytes"])
        or len(context.names) != int(expected_manifest["mapping_image_count"])
        or intersection != 0
    ):
        raise ValueError("P9 ordered source mapping images differ from preregistration")
    return {
        "manifest_path": expected_manifest["path"],
        "manifest_sha256": expected_manifest["sha256"],
        "ordered_source_image_manifest_sha256": digest.hexdigest(),
        "mapping_image_count": len(context.names),
        "mapping_image_total_bytes": total_bytes,
        "mapping_test_name_intersection_count": intersection,
    }


def implementation_registry() -> dict:
    """Require a later committed review/full-test registry for real runs."""
    if not IMPLEMENTATION_REGISTRY_PATH.is_file():
        raise RuntimeError(
            "Reviewed P9 implementation registry is not committed; real mapping "
            "execution is forbidden"
        )
    payload = json.loads(IMPLEMENTATION_REGISTRY_PATH.read_text())
    required_paths = sorted(PRODUCER_SOURCE_PATHS)
    root = Path(__file__).resolve().parents[1]
    if (
        payload.get("schema")
        != "lafgs_p9_fixed_pair_matcher_ceiling_implementation_registry"
        or payload.get("version") != 1
        or payload.get("valid") is not True
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("preregistration", {}).get("commit")
        != PREREGISTRATION_AMENDMENT_COMMIT
        or payload.get("preregistration", {}).get("blob_sha256")
        != PREREGISTRATION_BLOB_SHA256
        or payload.get("required_source_paths") != required_paths
        or payload.get("source_file_sha256")
        != {name: sha256_file(root / name) for name in required_paths}
        or payload.get("full_cpu_tests", {}).get("passed") is not True
        or payload.get("independent_review", {}).get("passed") is not True
        or payload.get("independent_review", {}).get("finding_counts")
        != {"p0": 0, "p1": 0, "p2": 0}
        or payload.get("authorizes_real_mapping_pair_gate") is not True
        or payload.get("authorizes_track") is not False
        or payload.get("authorizes_test") is not False
    ):
        raise RuntimeError("P9 implementation registry is invalid or stale")
    commit = str(payload.get("implementation_commit", ""))
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("P9 implementation registry lacks a commit identity")
    current = _git(root, "rev-parse", "HEAD")
    registry_relative = str(IMPLEMENTATION_REGISTRY_PATH.relative_to(root))
    prereg_relative = str(PREREGISTRATION_PATH.relative_to(root))
    try:
        committed_registry = subprocess.run(
            ["git", "-C", str(root), "show", f"{current}:{registry_relative}"],
            check=True,
            capture_output=True,
        ).stdout
        committed_preregistration = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{PREREGISTRATION_AMENDMENT_COMMIT}:{prereg_relative}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("P9 registry/preregistration is not committed") from error
    if (
        committed_registry != IMPLEMENTATION_REGISTRY_PATH.read_bytes()
        or hashlib.sha256(committed_preregistration).hexdigest()
        != PREREGISTRATION_BLOB_SHA256
    ):
        raise RuntimeError("P9 committed registry/preregistration blob differs")
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, current],
        check=False,
        capture_output=True,
    ).returncode:
        raise RuntimeError("Reviewed P9 implementation is not in current history")
    if subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            PREREGISTRATION_AMENDMENT_COMMIT,
            commit,
        ],
        check=False,
        capture_output=True,
    ).returncode:
        raise RuntimeError(
            "P9 preregistration amendment does not precede implementation"
        )
    return payload


def _git(worktree: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(worktree), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot verify Git lineage: {' '.join(arguments)}") from error


def _load_module(path: Path, *, label: str) -> ModuleType:
    source = path.read_bytes()
    name = f"_lafgs_p9_{label}_{hashlib.sha256(source).hexdigest()[:16]}"
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _expected_digest(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def validate_bundled_xfeat_artifact(spec: BundledXFeatSpec) -> dict:
    """Validate the exact clean external checkout and bundled 291-key file."""
    prereg = preregistration()
    expected = prereg["external_xfeat"]
    expected_checkpoint = prereg["checkpoint"]
    worktree = Path(spec.worktree).expanduser().resolve()
    checkpoint = Path(spec.checkpoint).expanduser().resolve()
    if not worktree.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("P9 XFeat worktree/checkpoint is missing")
    if Path(_git(worktree, "rev-parse", "--show-toplevel")).resolve() != worktree:
        raise ValueError("P9 XFeat worktree must be the Git top level")
    commit = _git(worktree, "rev-parse", "HEAD").lower()
    tree = _git(worktree, "rev-parse", "HEAD:encoders/XFeat").lower()
    if (
        commit != str(spec.expected_parent_commit).lower()
        or commit != expected["parent_commit"]
    ):
        raise ValueError("P9 external XFeat parent commit differs")
    if tree != str(spec.expected_xfeat_tree).lower() or tree != expected["xfeat_tree"]:
        raise ValueError("P9 external XFeat tree differs")
    if _git(worktree, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("P9 external XFeat worktree is not clean")
    expected_checkpoint_path = worktree / "encoders/XFeat/weights/xfeat-lighterglue.pt"
    if checkpoint != expected_checkpoint_path.resolve():
        raise ValueError("P9 accepts only the bundled xfeat-lighterglue.pt")
    checkpoint_sha = sha256_file(checkpoint)
    if (
        checkpoint_sha
        != _expected_digest(
            spec.expected_checkpoint_sha256,
            label="expected bundled checkpoint SHA-256",
        )
        or checkpoint_sha != expected_checkpoint["sha256"]
        or checkpoint.stat().st_size != expected_checkpoint["size_bytes"]
        or _git(
            worktree,
            "rev-parse",
            "HEAD:encoders/XFeat/weights/xfeat-lighterglue.pt",
        ).lower()
        != expected_checkpoint["git_blob"]
    ):
        raise ValueError("P9 bundled XFeat checkpoint identity differs")
    files = {}
    for label, contract in expected["files"].items():
        path = (worktree / contract["path"]).resolve()
        try:
            path.relative_to(worktree)
        except ValueError as error:
            raise ValueError("P9 XFeat source path escapes worktree") from error
        if not path.is_file() or sha256_file(path) != contract["sha256"]:
            raise ValueError(f"P9 XFeat {label} source identity differs")
        _git(worktree, "ls-files", "--error-unmatch", contract["path"])
        files[label] = {
            "path": str(path),
            "sha256": contract["sha256"],
            "git_blob": _git(worktree, "rev-parse", f"HEAD:{contract['path']}").lower(),
            "size_bytes": int(path.stat().st_size),
        }
    return {
        "worktree": str(worktree),
        "parent_commit": commit,
        "xfeat_tree": tree,
        "git_clean": True,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "git_blob": expected_checkpoint["git_blob"],
            "size_bytes": int(checkpoint.stat().st_size),
        },
        "files": files,
    }


def load_bundled_checkpoint(path: str | Path) -> OrderedDict[str, torch.Tensor]:
    """Safely load and structurally audit the exact bundled checkpoint."""
    try:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("cannot safely load bundled XFeat checkpoint") from error
    if not isinstance(state, Mapping):
        raise ValueError("bundled checkpoint must be a tensor mapping")
    ordered = OrderedDict(state.items())
    if len(ordered) != CHECKPOINT_KEY_COUNT or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in ordered.items()
    ):
        raise ValueError("bundled checkpoint is not the exact 291-tensor mapping")
    if any(
        (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all())
        for value in ordered.values()
    ):
        raise ValueError("bundled checkpoint contains non-finite tensors")
    keys = list(ordered)
    extractor_keys = [key for key in keys if key.startswith(EXTRACTOR_PREFIX)]
    matcher_keys = [key for key in keys if key.startswith(MATCHER_PREFIX)]
    if (
        _key_order_sha256(keys) != RAW_KEY_ORDER_SHA256
        or len(extractor_keys) != EXTRACTOR_KEY_COUNT
        or len(matcher_keys) != MATCHER_KEY_COUNT
        or len(extractor_keys) + len(matcher_keys) != len(keys)
        or _key_order_sha256(
            [key.removeprefix(EXTRACTOR_PREFIX) for key in extractor_keys]
        )
        != EXTRACTOR_KEY_ORDER_SHA256
        or _key_order_sha256([key.removeprefix(MATCHER_PREFIX) for key in matcher_keys])
        != MATCHER_KEY_ORDER_SHA256
    ):
        raise ValueError("bundled checkpoint key split/order differs")
    return ordered


def _rewrite_matcher_key(key: str, *, layers: int = 6) -> str:
    rewritten = key
    for index in range(layers):
        rewritten = rewritten.replace(
            f"self_attn.{index}", f"transformers.{index}.self_attn"
        )
        rewritten = rewritten.replace(
            f"cross_attn.{index}", f"transformers.{index}.cross_attn"
        )
    return rewritten


def strict_load_bundled_state(
    state: Mapping[str, torch.Tensor],
    *,
    extractor: torch.nn.Module,
    matcher: torch.nn.Module,
) -> dict:
    """Load E1 and LighterGlue without any compatibility fall-through."""
    ordered = OrderedDict(state.items())
    # The in-memory entry point is intentionally as strict as the file loader.
    keys = list(ordered)
    extractor_keys = [key for key in keys if key.startswith(EXTRACTOR_PREFIX)]
    matcher_keys = [key for key in keys if key.startswith(MATCHER_PREFIX)]
    if (
        len(keys) != CHECKPOINT_KEY_COUNT
        or _key_order_sha256(keys) != RAW_KEY_ORDER_SHA256
        or _key_order_sha256(
            [key.removeprefix(EXTRACTOR_PREFIX) for key in extractor_keys]
        )
        != EXTRACTOR_KEY_ORDER_SHA256
        or _key_order_sha256([key.removeprefix(MATCHER_PREFIX) for key in matcher_keys])
        != MATCHER_KEY_ORDER_SHA256
        or len(extractor_keys) != EXTRACTOR_KEY_COUNT
        or len(matcher_keys) != MATCHER_KEY_COUNT
    ):
        raise ValueError("bundled state cannot be split into exact E1/LG axes")
    extractor_state = OrderedDict(
        (key.removeprefix(EXTRACTOR_PREFIX), ordered[key]) for key in extractor_keys
    )
    try:
        incompatible = extractor.load_state_dict(extractor_state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "bundled E1 extractor state fails strict load (shape/BN axis mismatch)"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("bundled E1 strict load returned incompatible keys")

    runtime_state = matcher.state_dict()
    if set(runtime_state) - {"confidence_thresholds"} != {
        _rewrite_matcher_key(key.removeprefix(MATCHER_PREFIX)) for key in matcher_keys
    } or set(runtime_state) - {
        _rewrite_matcher_key(key.removeprefix(MATCHER_PREFIX)) for key in matcher_keys
    } != {"confidence_thresholds"}:
        raise ValueError("instantiated LighterGlue state is not 169+one runtime buffer")
    confidence = runtime_state["confidence_thresholds"].detach().cpu().contiguous()
    if (
        confidence.dtype != torch.float32
        or tuple(confidence.shape) != (6,)
        or _tensor_bytes_sha256(confidence) != CONFIDENCE_THRESHOLDS_SHA256
    ):
        raise ValueError("LighterGlue confidence_thresholds buffer differs")
    rewritten = {
        _rewrite_matcher_key(key.removeprefix(MATCHER_PREFIX)): ordered[key]
        for key in matcher_keys
    }
    final_state = OrderedDict()
    for key in runtime_state:
        final_state[key] = (
            confidence if key == "confidence_thresholds" else rewritten[key]
        )
    try:
        incompatible = matcher.load_state_dict(final_state, strict=True)
    except RuntimeError as error:
        raise ValueError("bundled LighterGlue state fails strict load") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("bundled LighterGlue strict load returned incompatible keys")
    return {
        "checkpoint_key_count": len(ordered),
        "extractor_key_count": len(extractor_state),
        "matcher_checkpoint_key_count": len(rewritten),
        "matcher_runtime_key_count": len(final_state),
        "runtime_only_buffer": "confidence_thresholds",
        "confidence_thresholds_sha256": CONFIDENCE_THRESHOLDS_SHA256,
        "extractor_strict_load": True,
        "matcher_strict_load": True,
        "strict_false_used": False,
    }


def load_bundled_models(
    artifact: Mapping,
) -> tuple[torch.nn.Module, torch.nn.Module, Any, dict]:
    """Instantiate exact E1, exact LighterGlue and the locked interpolator."""
    from kornia.feature.lightglue import LightGlue
    import inspect

    source = Path(inspect.getsourcefile(LightGlue) or "").resolve()
    runtime_contract = preregistration()["runtime"]
    license_path = Path(runtime_contract["kornia_license_path"]).resolve()
    if not source.is_file() or sha256_file(source) != LIGHTGLUE_SOURCE_SHA256:
        raise ValueError(
            "installed Kornia LightGlue source differs from preregistration"
        )
    if (
        not license_path.is_file()
        or sha256_file(license_path) != runtime_contract["kornia_license_sha256"]
    ):
        raise ValueError("installed Kornia license differs from preregistration")
    if (
        hashlib.sha256(canonical_json(LIGHTGLUE_CONFIG).encode("utf-8")).hexdigest()
        != LIGHTGLUE_CONFIG_SHA256
    ):
        raise RuntimeError("in-repo P9 LighterGlue config is stale")
    model_module = _load_module(Path(artifact["files"]["model"]["path"]), label="model")
    interpolator_module = _load_module(
        Path(artifact["files"]["interpolator"]["path"]), label="interpolator"
    )
    model_class = getattr(model_module, "XFeatModel", None)
    interpolator_class = getattr(interpolator_module, "InterpolateSparse2d", None)
    if model_class is None or interpolator_class is None:
        raise ValueError("locked XFeat source lacks required classes")
    extractor = model_class().cpu().eval()
    matcher = LightGlue(None, **LIGHTGLUE_CONFIG).cpu().eval()
    state = load_bundled_checkpoint(artifact["checkpoint"]["path"])
    summary = strict_load_bundled_state(state, extractor=extractor, matcher=matcher)
    interpolators = {
        mode: interpolator_class(mode).cpu().eval()
        for mode in ("nearest", "bilinear", "bicubic")
    }
    if any(
        str(getattr(interpolators[mode], "mode", "")) != mode for mode in interpolators
    ):
        raise ValueError("locked XFeat interpolator modes differ")
    for module in (extractor, matcher, *interpolators.values()):
        if module.training:
            raise RuntimeError("P9 model/interpolator escaped eval mode")
        if any(value.device.type != "cpu" for value in module.parameters()) or any(
            value.device.type != "cpu" for value in module.buffers()
        ):
            raise RuntimeError("P9 model escaped the CPU-only contract")
        if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in (*module.parameters(), *module.buffers())
        ):
            raise RuntimeError("P9 model escaped the float32 contract")
    return extractor, matcher, interpolators, summary


def _safe_name(name: str) -> str:
    value = str(name).replace("\\", "/")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe mapping image name: {name!r}")
    return value


def validate_mapping_context(
    query_cache: Mapping,
    *,
    query_cache_path: str | Path,
    expected_query_cache_sha256: str,
    dataset_root: str | Path,
    images: str,
    expected_query_names_sha256: str,
    expected_query_count: int,
    requested_keypoint_count: int,
    mapping_scope: Mapping,
    source_image_manifest: Mapping,
) -> MappingFeatureContext:
    """Bind cache records exactly to the dataset's mapping split."""
    path = Path(query_cache_path).expanduser().resolve()
    expected_cache_sha = _expected_digest(
        expected_query_cache_sha256, label="expected query-cache SHA-256"
    )
    if not path.is_file() or sha256_file(path) != expected_cache_sha:
        raise ValueError("P9 query-cache artifact differs")
    if query_cache.get("uses_test_queries") is True:
        raise ValueError("P9 feature cache cannot consume test queries")
    records = query_cache.get("queries", query_cache)
    signature = query_cache.get("signature_payload")
    if not isinstance(records, Mapping) or not isinstance(signature, Mapping):
        raise ValueError("P9 query cache lacks records/signature")
    dataset = _CPUColmapDataset(dataset_root, images=images)
    cameras = tuple(dataset.split("mapping"))
    names = tuple(_safe_name(camera.image_name) for camera in cameras)
    expected_names_sha = _expected_digest(
        expected_query_names_sha256, label="expected query-name SHA-256"
    )
    if (
        len(names) != int(expected_query_count)
        or tuple(records) != names
        or set(records) != set(names)
        or ordered_names_sha256(names) != expected_names_sha
    ):
        raise ValueError("P9 mapping image/cache registry differs")
    requested_k = int(requested_keypoint_count)
    if requested_k <= 0:
        raise ValueError("P9 requested keypoint count must be positive")
    if int(signature.get("native_sparse_keypoint_count", -1)) != requested_k:
        raise ValueError("P9 source-cache K differs from the scene registry")
    if signature.get("valid_mask_policy") != "object_and_sky_and_distortion_v1":
        raise ValueError("P9 source-cache mask policy differs")
    if (
        Path(str(signature.get("source_path", ""))).expanduser().resolve()
        != Path(dataset_root).expanduser().resolve()
    ):
        raise ValueError("P9 dataset root differs from query-cache lineage")
    if str(signature.get("images", "")) != str(images):
        raise ValueError("P9 images directory differs from query-cache lineage")
    if mapping_scope.get("uses_test_queries") is not False:
        raise ValueError("P9 mapping-scope proof does not exclude test queries")
    for name in names:
        record = records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"P9 query record is not a mapping: {name}")
        height, width = tuple(int(value) for value in record.get("native_input_hw", ()))
        if height <= 0 or width <= 0:
            raise ValueError(f"P9 query has invalid native dimensions: {name}")
        for field in (
            "native_depth",
            "native_alpha",
            "native_valid_mask",
            "native_K",
            "pose_w2c",
        ):
            if field not in record:
                raise ValueError(f"P9 query lacks dense geometry field {field}: {name}")
    return MappingFeatureContext(
        dataset=dataset,
        cameras=cameras,
        names=names,
        records=records,
        signature_payload=signature,
        mapping_scope=dict(mapping_scope),
        query_cache_path=path,
        query_cache_sha256=expected_cache_sha,
        query_names_sha256=expected_names_sha,
        requested_keypoint_count=requested_k,
        source_image_manifest=dict(source_image_manifest),
    )


def _mask_at_hw(dataset: ColmapDataset, name: str, hw: tuple[int, int]) -> torch.Tensor:
    masks = getattr(dataset, "_masks", None)
    if masks is None or name not in masks:
        return torch.ones(hw, dtype=torch.bool)
    channels = masks[name]
    if len(channels) < 3:
        raise ValueError(f"P9 valid mask lacks three channels: {name}")
    resized = []
    for channel in channels[:3]:
        value = torch.as_tensor(channel, dtype=torch.float32, device="cpu")
        if value.ndim != 2:
            raise ValueError(f"P9 valid-mask channel is not 2D: {name}")
        resized.append(
            F.interpolate(value[None, None], size=hw, mode="nearest")[0, 0].bool()
        )
    return resized[0] & resized[1] & resized[2]


def recreate_native_mapping_input(
    context: MappingFeatureContext,
    camera,
    cached: Mapping,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Recreate the exact masked RGB input, including non-/32 dimensions."""
    before_sha = sha256_file(camera.image_path)
    image = context.dataset.load_image(camera).cpu().float()
    if sha256_file(camera.image_path) != before_sha:
        raise RuntimeError(
            f"P9 mapping image changed while loading: {camera.image_name}"
        )
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("P9 mapping RGB must be CHW with one or three channels")
    original_hw = tuple(int(value) for value in image.shape[-2:])
    resolution = int(context.signature_payload.get("resolution", 0))
    if resolution <= 0:
        raise ValueError("P9 source resolution contract is invalid")
    scene_hw = original_hw
    if resolution != 1:
        scene_hw = (
            round(original_hw[0] / float(resolution)),
            round(original_hw[1] / float(resolution)),
        )
        image = F.interpolate(
            image[None], size=scene_hw, mode="bilinear", align_corners=False
        )[0]
    mask = _mask_at_hw(context.dataset, camera.image_name, scene_hw)
    native_hw = resolution_from_longest_edge(
        *scene_hw, int(context.signature_payload.get("longest_edge", 0))
    )
    if tuple(image.shape[-2:]) != tuple(native_hw):
        image = F.interpolate(
            image[None], size=native_hw, mode="bilinear", align_corners=False
        )[0]
        mask = F.interpolate(mask[None, None].float(), size=native_hw, mode="nearest")[
            0, 0
        ].bool()
    if tuple(native_hw) != tuple(int(value) for value in cached["native_input_hw"]):
        raise ValueError("P9 recreated native dimensions differ from source cache")
    cached_mask_value = torch.as_tensor(cached["native_valid_mask"])
    if cached_mask_value.ndim != 2:
        raise ValueError("P9 cached native valid mask must have exact shape [H,W]")
    cached_mask = cached_mask_value.bool()
    if not torch.equal(mask, cached_mask):
        raise ValueError("P9 recreated native valid mask differs from source cache")
    if (
        not bool(torch.isfinite(image).all())
        or bool((image < 0).any())
        or bool((image > 1).any())
    ):
        raise ValueError("P9 recreated RGB is outside finite [0,1]")
    masked = (image * mask[None].to(image.dtype)).contiguous()
    return (
        masked,
        mask.contiguous(),
        {
            "source_image_path": str(Path(camera.image_path).resolve()),
            "source_image_sha256": before_sha,
            "source_camera_hw": list(original_hw),
            "scene_resolution_hw": list(scene_hw),
            "native_input_hw": list(native_hw),
            "native_valid_mask_sha256": tensor_sha256(mask),
            "native_masked_rgb_sha256": tensor_sha256(masked),
            "rgb_resize_mode": "bilinear_align_corners_false",
            "mask_resize_mode": "nearest",
            "mask_policy": "object_and_sky_and_distortion_v1",
        },
    )


def xfeat_resize_contract(native_hw: Sequence[int]) -> dict:
    height, width = (int(value) for value in native_hw)
    xfeat_height, xfeat_width = (height // 32) * 32, (width // 32) * 32
    if min(xfeat_height, xfeat_width) < 32:
        raise ValueError("P9 XFeat input dimensions must be at least 32")
    return {
        "native_input_hw": [height, width],
        "xfeat_input_hw": [xfeat_height, xfeat_width],
        "rh": float(height / xfeat_height),
        "rw": float(width / xfeat_width),
        "resize_mode": "bilinear_align_corners_false",
        "raw_to_native_xy": "(rw*x,rh*y)",
    }


def _keypoint_heatmap(logits: torch.Tensor) -> torch.Tensor:
    logits = torch.as_tensor(logits).detach().cpu().float()
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 65:
        raise ValueError("P9 E1 logits must have shape [1,65,H,W]")
    probabilities = F.softmax(logits, dim=1)[:, :64]
    batch, _, height, width = probabilities.shape
    heatmap = probabilities.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
    return heatmap.permute(0, 1, 3, 2, 4).reshape(batch, 1, height * 8, width * 8)


def extract_bundled_features(
    *,
    extractor: torch.nn.Module,
    interpolators: Mapping[str, torch.nn.Module],
    native_image: torch.Tensor,
    valid_mask: torch.Tensor,
    requested_keypoint_count: int,
) -> dict:
    """Run exactly one E1 forward and materialize fresh detector rows."""
    native_image = torch.as_tensor(native_image)
    valid_mask = torch.as_tensor(valid_mask)
    if (
        native_image.ndim != 3
        or native_image.shape[0] not in (1, 3)
        or valid_mask.dtype != torch.bool
        or tuple(valid_mask.shape) != tuple(native_image.shape[-2:])
    ):
        raise ValueError("P9 native image/mask tensors have invalid exact shapes")
    contract = xfeat_resize_contract(native_image.shape[-2:])
    xfeat_hw = tuple(contract["xfeat_input_hw"])
    model_input = F.interpolate(
        torch.as_tensor(native_image).cpu().float()[None],
        size=xfeat_hw,
        mode="bilinear",
        align_corners=False,
    )
    with torch.inference_mode():
        outputs = extractor(model_input)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError("P9 E1 must return descriptor/logit/reliability outputs")
    dense, logits, reliability = (
        torch.as_tensor(value).detach().cpu().float() for value in outputs
    )
    expected_grid = (xfeat_hw[0] // 8, xfeat_hw[1] // 8)
    if (
        tuple(dense.shape) != (1, DESCRIPTOR_DIM, *expected_grid)
        or tuple(logits.shape) != (1, 65, *expected_grid)
        or tuple(reliability.shape) != (1, 1, *expected_grid)
        or not all(bool(torch.isfinite(value).all()) for value in outputs)
    ):
        raise ValueError("P9 E1 output shapes/values differ")
    dense = F.normalize(dense, dim=1)
    heatmap = _keypoint_heatmap(logits)
    local = F.max_pool2d(
        heatmap,
        kernel_size=NMS_KERNEL_SIZE,
        stride=1,
        padding=NMS_KERNEL_SIZE // 2,
    )
    selected = (heatmap == local) & (heatmap > DETECTION_THRESHOLD)
    yx = selected[0, 0].nonzero(as_tuple=False)
    raw_xy = yx[:, [1, 0]].float().contiguous()
    detected_after_nms = int(raw_xy.shape[0])
    if raw_xy.numel():
        positions = raw_xy[None]
        probability_value = torch.as_tensor(
            interpolators["nearest"](heatmap, positions, H=xfeat_hw[0], W=xfeat_hw[1])
        )
        reliability_value = torch.as_tensor(
            interpolators["bilinear"](
                reliability, positions, H=xfeat_hw[0], W=xfeat_hw[1]
            )
        )
        expected_score_shape = (1, detected_after_nms, 1)
        if (
            tuple(probability_value.shape) != expected_score_shape
            or tuple(reliability_value.shape) != expected_score_shape
        ):
            raise ValueError(
                "P9 interpolated detector scores have invalid exact shapes"
            )
        probability = probability_value[0, :, 0].float()
        reliability_score = reliability_value[0, :, 0].float()
        score = probability * reliability_score
        score[torch.all(raw_xy == 0, dim=1)] = -1.0
        order = torch.argsort(-score, stable=True)[: int(requested_keypoint_count)]
        raw_xy, score = raw_xy[order], score[order]
        positive = score > 0
        raw_xy, score = raw_xy[positive], score[positive]
    else:
        score = torch.empty(0, dtype=torch.float32)
    topk_before_mask_count = int(raw_xy.shape[0])
    scale = raw_xy.new_tensor([contract["rw"], contract["rh"]])
    native_xy = (raw_xy * scale).contiguous()
    keep = sample_mask_at_grid_uv(valid_mask, native_xy)
    raw_xy = raw_xy[keep].contiguous()
    native_xy = native_xy[keep].contiguous()
    score = score[keep].contiguous()
    if raw_xy.numel():
        descriptors = interpolators["bicubic"](
            dense,
            raw_xy[None],
            H=xfeat_hw[0],
            W=xfeat_hw[1],
        )[0]
        descriptors = F.normalize(descriptors.float(), dim=1).contiguous()
    else:
        descriptors = torch.empty((0, DESCRIPTOR_DIM), dtype=torch.float32)
    if (
        descriptors.shape != (native_xy.shape[0], DESCRIPTOR_DIM)
        or score.shape != (native_xy.shape[0],)
        or not all(
            bool(torch.isfinite(value).all())
            for value in (raw_xy, native_xy, descriptors, score)
        )
    ):
        raise ValueError("P9 E1 feature rows are invalid")
    return {
        "raw_xfeat_xy": raw_xy,
        "native_xy": native_xy,
        "descriptor": descriptors,
        "detector_score": score,
        "detector_lineage": {
            **contract,
            "single_model_forward": True,
            "detection_threshold_strict_greater_than": DETECTION_THRESHOLD,
            "nms_kernel_size": NMS_KERNEL_SIZE,
            "nms_radius": NMS_KERNEL_SIZE // 2,
            "nms_passes": 1,
            "sort": "descending_score_stable_row_major_ties",
            "requested_top_k_before_mask": int(requested_keypoint_count),
            "mask_refill": False,
            "candidate_count_after_threshold_nms": detected_after_nms,
            "positive_top_k_count_before_mask": topk_before_mask_count,
            "post_mask_count": int(native_xy.shape[0]),
            "raw_coordinates_preserved": True,
            "native_coordinates_scaled": True,
        },
    }


def resample_dense_teacher(
    cached: Mapping, native_hw: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Resample source dense depth/alpha onto the exact declared native grid."""
    target_hw = tuple(int(value) for value in native_hw)
    depth_value = torch.as_tensor(cached["native_depth"])
    alpha_value = torch.as_tensor(cached["native_alpha"])
    if depth_value.ndim != 2 or alpha_value.ndim != 2:
        raise ValueError("P9 dense teacher fields must have exact shape [H,W]")
    depth = depth_value.detach().cpu().float()
    alpha = alpha_value.detach().cpu().float()
    source_depth_hw = tuple(depth.shape)
    source_alpha_hw = tuple(alpha.shape)
    if source_depth_hw != target_hw:
        depth = F.interpolate(
            depth[None, None], size=target_hw, mode="bilinear", align_corners=False
        )[0, 0]
    if source_alpha_hw != target_hw:
        alpha = F.interpolate(
            alpha[None, None], size=target_hw, mode="bilinear", align_corners=False
        )[0, 0]
    alpha = alpha.clamp(0.0, 1.0)
    if not bool(torch.isfinite(depth).all()) or not bool(torch.isfinite(alpha).all()):
        raise ValueError("P9 dense teacher contains non-finite values")
    return (
        depth.contiguous(),
        alpha.to(torch.float16).contiguous(),
        {
            "source_native_depth_hw": list(source_depth_hw),
            "source_native_alpha_hw": list(source_alpha_hw),
            "target_native_hw": list(target_hw),
            "depth_resample": "identity_or_bilinear_align_corners_false",
            "alpha_resample": "identity_or_bilinear_align_corners_false",
            "alpha_threshold": ALPHA_THRESHOLD,
        },
    )


def _query_hashes(record: Mapping) -> dict[str, str]:
    names = (
        "raw_xfeat_xy",
        "native_xy",
        "descriptor",
        "detector_score",
        "native_depth_resampled",
        "native_alpha_resampled",
        "native_valid_mask",
        "native_K",
        "pose_w2c",
    )
    hashes = {name: tensor_sha256(torch.as_tensor(record[name])) for name in names}
    hashes["row_registry_sha256"] = hashlib.sha256(
        canonical_json(
            {
                "raw_xfeat_xy": hashes["raw_xfeat_xy"],
                "native_xy": hashes["native_xy"],
                "descriptor": hashes["descriptor"],
                "detector_score": hashes["detector_score"],
                "row_count": int(torch.as_tensor(record["native_xy"]).shape[0]),
            }
        ).encode("utf-8")
    ).hexdigest()
    return hashes


def feature_cache_content_sha256(payload: Mapping) -> str:
    queries = payload.get("queries", {})
    summary = {
        key: value
        for key, value in payload.items()
        if key not in {"queries", "content_sha256"}
    }
    tensor_fields = {
        "raw_xfeat_xy",
        "native_xy",
        "descriptor",
        "detector_score",
        "native_depth_resampled",
        "native_alpha_resampled",
        "native_valid_mask",
        "native_K",
        "pose_w2c",
    }
    summary["queries"] = [
        {
            "name": name,
            **{key: value for key, value in record.items() if key not in tensor_fields},
        }
        for name, record in queries.items()
    ]
    return hashlib.sha256(canonical_json(summary).encode("utf-8")).hexdigest()


def validate_feature_cache(
    payload: Mapping,
    *,
    expected_scene: str | None = None,
    expected_content_sha256: str | None = None,
) -> dict:
    """Recompute every scientific field hash and validate cache structure."""
    schema_contract = preregistration()["artifact_schemas"]["feature_cache"]
    if (
        not set(schema_contract["required_top_level_keys"]).issubset(payload)
        or payload.get("schema") != FEATURE_CACHE_SCHEMA
        or payload.get("version") != FEATURE_CACHE_VERSION
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("scene") not in {"stairs", "greatcourt"}
        or re.fullmatch(r"[0-9a-f]{32}", str(payload.get("run_uuid", ""))) is None
        or not isinstance(payload.get("producer_identity"), Mapping)
        or SHA256_PATTERN.fullmatch(
            str(payload.get("producer_identity", {}).get("compiled_identity", ""))
        )
        is None
    ):
        raise ValueError("unexpected P9 feature-cache schema/scope")
    if expected_scene is not None and payload.get("scene") != expected_scene:
        raise ValueError("P9 feature cache names a different scene")
    names = payload.get("query_names")
    queries = payload.get("queries")
    if (
        not isinstance(names, list)
        or not isinstance(queries, Mapping)
        or list(queries) != names
        or len(names) != int(payload.get("query_count", -1))
        or ordered_names_sha256(names) != payload.get("query_names_sha256")
    ):
        raise ValueError("P9 feature-cache query registry is invalid")
    requested_k = int(payload.get("requested_keypoint_count", -1))
    scene_contract = preregistration()["fixed_scene_registry"][payload["scene"]]
    scene_source_contract = scene_contract["mapping_source_image_manifest"]
    source_inputs = payload.get("inputs", {}).get("mapping_source_images", {})
    parent = payload.get("inputs", {}).get("parent_stairs_gate")
    query_cache = payload.get("inputs", {}).get("query_cache", {})
    extractor = payload.get("extractor", {})
    state = extractor.get("state", {})
    feature_contract = payload.get("feature_contract", {})
    expected_manifest_path = str(
        (Path(__file__).resolve().parents[1] / scene_source_contract["path"]).resolve()
    )
    if (
        requested_k <= 0
        or source_inputs.get("manifest_path") != expected_manifest_path
        or source_inputs.get("manifest_sha256") != scene_source_contract["sha256"]
        or source_inputs.get("ordered_source_image_manifest_sha256")
        != scene_source_contract["ordered_source_image_manifest_sha256"]
        or source_inputs.get("mapping_image_count")
        != scene_source_contract["mapping_image_count"]
        or source_inputs.get("mapping_image_total_bytes")
        != scene_source_contract["mapping_image_total_bytes"]
        or source_inputs.get("mapping_test_name_intersection_count") != 0
        or query_cache.get("sha256") != scene_contract["query_cache"]["sha256"]
        or query_cache.get("mapping_scope", {}).get("uses_test_queries") is not False
        or extractor.get("name") != "E1_bundled_xfeat_extractor"
        or extractor.get("checkpoint", {}).get("path")
        != preregistration()["checkpoint"]["path"]
        or extractor.get("checkpoint", {}).get("sha256")
        != preregistration()["checkpoint"]["sha256"]
        or extractor.get("external_parent_commit")
        != preregistration()["external_xfeat"]["parent_commit"]
        or extractor.get("xfeat_tree")
        != preregistration()["external_xfeat"]["xfeat_tree"]
        or extractor.get("model", {}).get("sha256")
        != preregistration()["external_xfeat"]["files"]["model"]["sha256"]
        or extractor.get("interpolator", {}).get("sha256")
        != preregistration()["external_xfeat"]["files"]["interpolator"]["sha256"]
        or state.get("checkpoint_key_count") != CHECKPOINT_KEY_COUNT
        or state.get("extractor_key_count") != EXTRACTOR_KEY_COUNT
        or state.get("matcher_checkpoint_key_count") != MATCHER_KEY_COUNT
        or state.get("matcher_runtime_key_count") != MATCHER_KEY_COUNT + 1
        or state.get("extractor_strict_load") is not True
        or state.get("matcher_strict_load") is not True
        or state.get("strict_false_used") is not False
        or state.get("runtime_only_buffer") != "confidence_thresholds"
        or state.get("confidence_thresholds_sha256") != CONFIDENCE_THRESHOLDS_SHA256
        or feature_contract.get("single_forward_per_image") is not True
        or feature_contract.get("detection_threshold") != DETECTION_THRESHOLD
        or feature_contract.get("nms_kernel_size") != NMS_KERNEL_SIZE
        or feature_contract.get("nms_radius") != NMS_KERNEL_SIZE // 2
        or feature_contract.get("top_k_before_mask") is not True
        or feature_contract.get("mask_refill") is not False
        or feature_contract.get("raw_and_native_coordinates_stored") is not True
        or feature_contract.get("dense_depth_alpha_resampled") is not True
        or feature_contract.get("greatcourt_non_divisible_by_32_supported") is not True
    ):
        raise ValueError("P9 feature-cache extractor/input contract differs")
    if payload["scene"] == "stairs" and parent is not None:
        raise ValueError("P9 Stairs feature cache cannot have a parent gate")
    if payload["scene"] == "greatcourt" and (
        not isinstance(parent, Mapping)
        or not isinstance(parent.get("path"), str)
        or SHA256_PATTERN.fullmatch(str(parent.get("sha256", ""))) is None
        or parent.get("scientific_projection", {}).get("scene") != "stairs"
        or parent.get("scientific_projection", {}).get("scene_pair_gate_passed")
        is not True
        or parent.get("scientific_projection", {}).get("decision")
        != "SCENE_PAIR_PASS_REQUIRES_OTHER_SCENE"
        or parent.get("scientific_projection", {}).get("compiled_identity")
        != payload["producer_identity"].get("compiled_identity")
    ):
        raise ValueError("P9 GreatCourt feature cache lacks its Stairs parent gate")
    total_rows = 0
    for index, name in enumerate(names):
        record = queries[name]
        if (
            not set(schema_contract["query_required_keys"]).issubset(record)
            or record.get("query_index") != index
            or record.get("query_name") != name
        ):
            raise ValueError("P9 feature-cache query index/name mismatch")
        native_hw = tuple(int(value) for value in record.get("native_input_hw", ()))
        raw_value = torch.as_tensor(record.get("raw_xfeat_xy"))
        native_value = torch.as_tensor(record.get("native_xy"))
        descriptor_value = torch.as_tensor(record.get("descriptor"))
        score_value = torch.as_tensor(record.get("detector_score"))
        depth_value = torch.as_tensor(record.get("native_depth_resampled"))
        alpha_value = torch.as_tensor(record.get("native_alpha_resampled"))
        mask_value = torch.as_tensor(record.get("native_valid_mask"))
        camera_k_value = torch.as_tensor(record.get("native_K"))
        pose_value = torch.as_tensor(record.get("pose_w2c"))
        if (
            raw_value.dtype != torch.float32
            or native_value.dtype != torch.float32
            or descriptor_value.dtype != torch.float32
            or score_value.dtype != torch.float32
            or depth_value.dtype != torch.float32
            or alpha_value.dtype != torch.float16
            or mask_value.dtype != torch.bool
            or camera_k_value.dtype != torch.float32
            or pose_value.dtype != torch.float32
        ):
            raise ValueError(f"P9 feature-cache tensor dtype differs: {name}")
        raw = raw_value
        native = native_value
        descriptor = descriptor_value
        score = score_value
        depth = depth_value
        alpha = alpha_value
        mask = mask_value
        camera_k = camera_k_value
        pose = pose_value
        row_count = int(record.get("row_count", -1))
        if (
            type(record.get("row_count")) is not int
            or len(native_hw) != 2
            or min(native_hw) <= 0
            or raw.shape != (row_count, 2)
            or native.shape != (row_count, 2)
            or descriptor.shape != (row_count, DESCRIPTOR_DIM)
            or score.shape != (row_count,)
            or row_count <= 0
            or row_count > requested_k
            or depth.shape != native_hw
            or alpha.shape != native_hw
            or mask.shape != native_hw
            or camera_k.shape != (3, 3)
            or pose.shape not in {(3, 4), (4, 4)}
            or tuple(record.get("xfeat_input_hw", ()))
            != ((native_hw[0] // 32) * 32, (native_hw[1] // 32) * 32)
            or (payload["scene"] == "greatcourt" and native_hw != (1080, 1920))
        ):
            raise ValueError(f"P9 feature-cache tensors are misaligned: {name}")
        lineage = record.get("detector_lineage", {})
        xfeat_hw = tuple(int(value) for value in record["xfeat_input_hw"])
        scale = raw.new_tensor([native_hw[1] / xfeat_hw[1], native_hw[0] / xfeat_hw[0]])
        if (
            lineage.get("single_model_forward") is not True
            or lineage.get("detection_threshold_strict_greater_than")
            != DETECTION_THRESHOLD
            or lineage.get("nms_kernel_size") != NMS_KERNEL_SIZE
            or lineage.get("sort") != "descending_score_stable_row_major_ties"
            or lineage.get("mask_refill") is not False
            or lineage.get("native_input_hw") != list(native_hw)
            or lineage.get("xfeat_input_hw") != list(xfeat_hw)
            or lineage.get("requested_top_k_before_mask") != requested_k
            or lineage.get("post_mask_count") != row_count
            or lineage.get("raw_coordinates_preserved") is not True
            or lineage.get("native_coordinates_scaled") is not True
            or not torch.equal(native, raw * scale)
        ):
            raise ValueError(f"P9 feature-cache detector semantics differ: {name}")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (raw, native, descriptor, score, depth, alpha, camera_k, pose)
        ):
            raise ValueError(f"P9 feature-cache tensor is non-finite: {name}")
        if row_count and (
            bool((native[:, 0] < 0).any())
            or bool((native[:, 0] > native_hw[1] - 1).any())
            or bool((native[:, 1] < 0).any())
            or bool((native[:, 1] > native_hw[0] - 1).any())
            or not bool(sample_mask_at_grid_uv(mask, native).all())
            or bool((score <= 0).any())
            or bool(torch.all(raw == 0, dim=1).any())
            or not bool(
                torch.allclose(
                    torch.linalg.norm(descriptor, dim=1),
                    torch.ones(row_count),
                    atol=1e-5,
                    rtol=1e-5,
                )
            )
        ):
            raise ValueError(f"P9 feature rows escape native mask/image: {name}")
        image_lineage = record.get("image_lineage", {})
        dense_lineage = record.get("dense_teacher_lineage", {})
        if (
            image_lineage.get("native_input_hw") != list(native_hw)
            or image_lineage.get("native_valid_mask_sha256")
            != tensor_sha256(mask_value)
            or SHA256_PATTERN.fullmatch(
                str(image_lineage.get("source_image_sha256", ""))
            )
            is None
            or SHA256_PATTERN.fullmatch(
                str(image_lineage.get("native_masked_rgb_sha256", ""))
            )
            is None
            or dense_lineage.get("target_native_hw") != list(native_hw)
            or dense_lineage.get("alpha_threshold") != ALPHA_THRESHOLD
            or dense_lineage.get("depth_resample")
            != "identity_or_bilinear_align_corners_false"
            or dense_lineage.get("alpha_resample")
            != "identity_or_bilinear_align_corners_false"
            or bool((alpha < 0).any())
            or bool((alpha > 1).any())
        ):
            raise ValueError(f"P9 feature-cache image/depth lineage differs: {name}")
        observed_hashes = _query_hashes(record)
        if record.get("hashes") != observed_hashes:
            raise ValueError(f"P9 feature-cache row/geometry hash is stale: {name}")
        total_rows += row_count
    if (
        int(payload.get("producer", {}).get("total_feature_rows", -1)) != total_rows
        or payload.get("producer", {}).get("model_forward_count") != len(names)
        or payload.get("producer", {}).get("query_count") != len(names)
        or payload.get("producer", {}).get("device") != "cpu"
        or payload.get("producer", {}).get("gpu_used") is not False
    ):
        raise ValueError("P9 feature-cache total row count is stale")
    content = feature_cache_content_sha256(payload)
    if payload.get("content_sha256") != content or (
        expected_content_sha256 is not None
        and content
        != _expected_digest(
            expected_content_sha256, label="expected feature-cache content SHA-256"
        )
    ):
        raise ValueError("P9 feature-cache content SHA-256 is stale")
    return {
        "scene": payload["scene"],
        "query_count": len(names),
        "total_feature_rows": total_rows,
        "content_sha256": content,
    }


def materialize_feature_cache(
    *,
    scene: str,
    context: MappingFeatureContext,
    artifact: Mapping,
    extractor: torch.nn.Module,
    interpolators: Mapping[str, torch.nn.Module],
    state_summary: Mapping,
    producer_identity: Mapping,
    run_uuid: str,
    parent_stairs_gate: Mapping | None,
) -> dict:
    """Build one complete in-memory P9 feature cache."""
    source_image_validation = validate_source_images(
        context=context, expected_manifest=context.source_image_manifest
    )
    queries: OrderedDict[str, dict] = OrderedDict()
    total_rows = 0
    for index, (name, camera) in enumerate(zip(context.names, context.cameras)):
        cached = context.records[name]
        image, mask, image_lineage = recreate_native_mapping_input(
            context, camera, cached
        )
        features = extract_bundled_features(
            extractor=extractor,
            interpolators=interpolators,
            native_image=image,
            valid_mask=mask,
            requested_keypoint_count=context.requested_keypoint_count,
        )
        depth, alpha, dense_lineage = resample_dense_teacher(cached, image.shape[-2:])
        native_k = (
            torch.as_tensor(cached["native_K"]).detach().cpu().float().contiguous()
        )
        pose = torch.as_tensor(cached["pose_w2c"]).detach().cpu().float().contiguous()
        record = {
            "query_index": index,
            "query_name": name,
            "native_input_hw": list(image.shape[-2:]),
            "xfeat_input_hw": features["detector_lineage"]["xfeat_input_hw"],
            "row_count": int(features["native_xy"].shape[0]),
            "raw_xfeat_xy": features["raw_xfeat_xy"],
            "native_xy": features["native_xy"],
            "descriptor": features["descriptor"],
            "detector_score": features["detector_score"],
            "native_depth_resampled": depth,
            "native_alpha_resampled": alpha,
            "native_valid_mask": mask,
            "native_K": native_k,
            "pose_w2c": pose,
            "image_lineage": image_lineage,
            "detector_lineage": features["detector_lineage"],
            "dense_teacher_lineage": dense_lineage,
        }
        record["hashes"] = _query_hashes(record)
        queries[name] = record
        total_rows += record["row_count"]
    payload = {
        "schema": FEATURE_CACHE_SCHEMA,
        "version": FEATURE_CACHE_VERSION,
        "scene": str(scene),
        "mapping_only": True,
        "uses_test_queries": False,
        "run_uuid": str(run_uuid),
        "query_count": len(context.names),
        "query_names": list(context.names),
        "query_names_sha256": context.query_names_sha256,
        "requested_keypoint_count": context.requested_keypoint_count,
        "inputs": {
            "query_cache": {
                "path": str(context.query_cache_path),
                "sha256": context.query_cache_sha256,
                "mapping_scope": dict(context.mapping_scope),
            },
            "mapping_source_images": source_image_validation,
            "parent_stairs_gate": (
                dict(parent_stairs_gate) if parent_stairs_gate is not None else None
            ),
        },
        "extractor": {
            "name": "E1_bundled_xfeat_extractor",
            "checkpoint": dict(artifact["checkpoint"]),
            "external_parent_commit": artifact["parent_commit"],
            "xfeat_tree": artifact["xfeat_tree"],
            "model": dict(artifact["files"]["model"]),
            "interpolator": dict(artifact["files"]["interpolator"]),
            "state": dict(state_summary),
        },
        "feature_contract": {
            "single_forward_per_image": True,
            "detection_threshold": DETECTION_THRESHOLD,
            "nms_kernel_size": NMS_KERNEL_SIZE,
            "nms_radius": NMS_KERNEL_SIZE // 2,
            "top_k_before_mask": True,
            "mask_refill": False,
            "raw_and_native_coordinates_stored": True,
            "dense_depth_alpha_resampled": True,
            "greatcourt_non_divisible_by_32_supported": True,
        },
        "producer_identity": dict(producer_identity),
        "producer": {
            "device": "cpu",
            "gpu_used": False,
            "network_access_used": False,
            "query_count": len(context.names),
            "model_forward_count": len(context.names),
            "total_feature_rows": total_rows,
        },
        "queries": queries,
    }
    payload["content_sha256"] = feature_cache_content_sha256(payload)
    validate_feature_cache(payload, expected_scene=scene)
    final_source_image_validation = validate_source_images(
        context=context, expected_manifest=context.source_image_manifest
    )
    if final_source_image_validation != source_image_validation:
        raise RuntimeError("P9 mapping source images changed during feature extraction")
    return payload
