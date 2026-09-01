#!/usr/bin/env python3
"""Materialize sparse Gaussian geometry support for V21 adaptation queries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch

from common.hashing import sha256_file
from map_learning.v21_gaussian_support import (
    EVIDENCE_SEMANTICS,
    ROLE,
    SCHEMA,
    VERSION,
    atomic_torch_save_fresh,
    build_support_record,
    sample_keypoint_raster_support,
    sha256_json,
)
from map_learning.v21_test_cache import (
    validate_cache_payload,
    validate_shard_registry,
    validate_split_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_SOURCES = (
    "map_learning/v21_gaussian_support.py",
    "map_learning/v21_test_cache.py",
    "map_learning/v21_test_protocol.py",
    "priors/models.py",
    "priors/rendering.py",
    "scripts/materialize_v21_gaussian_support.py",
)


def _source(path: str | Path) -> dict[str, str | int]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _verify_sources(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        path = Path(str(source["path"]))
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(source["size_bytes"])
            or sha256_file(path) != source["sha256"]
        ):
            raise RuntimeError(f"V21 Gaussian source changed while running: {path}")


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("V21 Gaussian split manifest must be a JSON mapping")
    return value


def _frontend_cache_set(paths: list[Path]) -> tuple[list[dict], dict, list[dict]]:
    if not paths:
        raise ValueError("V21 Gaussian support requires frontend cache shards")
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("V21 Gaussian frontend cache paths are duplicated")
    loaded = []
    sources = []
    for path in resolved:
        source = _source(path)
        cache = torch.load(path, map_location="cpu", weights_only=False)
        validate_cache_payload(cache)
        if (
            cache.get("role") != ROLE
            or cache.get("training_consumers_allowed") is not True
            or cache.get("training_consumer_allowed") is not True
        ):
            raise ValueError(
                "V21 Gaussian support may only consume adaptation cache shards"
            )
        loaded.append(cache)
        sources.append(source)

    first = loaded[0]
    registry = first["shard_registry"]
    validate_shard_registry(registry)
    shard_count = int(first["shard_count"])
    by_shard: dict[int, tuple[dict, dict]] = {}
    invariant_fields = (
        "split_manifest_sha256",
        "anchor_count",
        "descriptor_dim",
        "role_query_count",
        "preprocessing_config_sha256",
        "frontend_contract",
        "baseline_contract",
    )
    for cache, source in zip(loaded, sources):
        if any(cache.get(field) != first.get(field) for field in invariant_fields):
            raise ValueError("V21 Gaussian frontend cache shard contracts differ")
        if cache.get("shard_registry") != registry:
            raise ValueError("V21 Gaussian frontend shard registries differ")
        shard = int(cache["shard_index"])
        if int(cache["shard_count"]) != shard_count or shard in by_shard:
            raise ValueError("V21 Gaussian frontend shard coordinate is duplicated")
        by_shard[shard] = (cache, source)
    if sorted(by_shard) != list(range(shard_count)):
        raise ValueError(
            "V21 Gaussian support requires complete frontend shard coverage"
        )

    records_by_query: dict[int, tuple[dict, dict, int]] = {}
    for shard in range(shard_count):
        cache, source = by_shard[shard]
        for record in cache["records"]:
            query = int(record["query_index"])
            if query in records_by_query:
                raise ValueError("V21 Gaussian frontend caches repeat a query")
            records_by_query[query] = (record, source, shard)
    ordered = []
    for row in sorted(registry["rows"], key=lambda value: int(value["ordinal"])):
        query = int(row["query_index"])
        item = records_by_query.get(query)
        if item is None:
            raise ValueError("V21 Gaussian frontend cache coverage is incomplete")
        record, source, shard = item
        if (
            shard != int(row["shard_index"])
            or record["image_name"] != row["image_name"]
            or record["image_sha256"] != row["image_sha256"]
            or record["source_record_sha256"] != row["source_record_sha256"]
        ):
            raise ValueError("V21 Gaussian frontend record registry differs")
        ordered.append({"record": record, "source": source, "shard_index": shard})
    if len(ordered) != int(registry["role_query_count"]):
        raise ValueError("V21 Gaussian frontend role coverage is incomplete")
    return loaded, registry, ordered


def _render_support(
    *,
    model: object,
    record: dict,
    render_fn: Callable[..., dict],
    device: torch.device,
    render_contract: dict,
) -> dict:
    height, width = map(int, torch.as_tensor(record["image_hw"]).tolist())
    intrinsic = torch.as_tensor(record["intrinsics"]).float()
    if intrinsic.shape != (3, 3):
        raise ValueError("V21 Gaussian frontend intrinsics are invalid")
    fov_x = 2.0 * math.atan(width / (2.0 * float(intrinsic[0, 0])))
    fov_y = 2.0 * math.atan(height / (2.0 * float(intrinsic[1, 1])))
    package = render_fn(
        model,
        torch.as_tensor(record["pose_w2c"], device=device, dtype=torch.float32),
        fov_x,
        fov_y,
        width,
        height,
        bg_color=torch.zeros(3, device=device),
        render_mode=render_contract["render_mode"],
        rgb_only=render_contract["rgb_only"],
    )
    if not isinstance(package, dict) or package.get("depth") is None:
        raise ValueError("V21 Gaussian renderer did not return depth")
    alpha = package.get("alphas", package.get("rend_alpha"))
    if alpha is None:
        raise ValueError("V21 Gaussian renderer did not return alpha")
    return sample_keypoint_raster_support(
        keypoints=record["keypoints"],
        depth=package["depth"],
        alpha=alpha,
        image_hw=(height, width),
        pixel_center_offset=float(render_contract["pixel_center_offset"]),
    )


@torch.inference_mode()
def materialize(
    args: argparse.Namespace,
    *,
    gaussian_factory: Callable[..., object] | None = None,
    render_fn: Callable[..., dict] | None = None,
) -> dict:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 Gaussian support output exists: {output}")
    split_path = Path(args.split_manifest).expanduser().resolve()
    map_path = Path(args.stable_map).expanduser().resolve()
    gaussian_path = Path(args.gaussian_ply).expanduser().resolve()
    split_source = _source(split_path)
    map_source = _source(map_path)
    gaussian_source = _source(gaussian_path)
    manifest = _load_json(split_path)
    adaptation_records = validate_split_manifest(manifest, role=ROLE)
    if (
        manifest.get("stable_map_sha256") != map_source["sha256"]
        or Path(str(manifest.get("stable_map", ""))).expanduser().resolve()
        != map_path
    ):
        raise ValueError("V21 Gaussian split is not bound to the stable map")

    caches, registry, ordered = _frontend_cache_set(
        [Path(path) for path in args.frontend_cache]
    )
    if (
        registry.get("split_manifest_sha256") != split_source["sha256"]
        or caches[0].get("split_manifest_sha256") != split_source["sha256"]
    ):
        raise ValueError("V21 Gaussian frontend caches use another split")
    cache_map = caches[0].get("inputs", {}).get("stable_map", {})
    if (
        cache_map.get("sha256") != map_source["sha256"]
        or Path(str(cache_map.get("path", ""))).expanduser().resolve() != map_path
    ):
        raise ValueError("V21 Gaussian frontend caches use another stable map")
    if len(adaptation_records) != len(ordered):
        raise ValueError("V21 Gaussian split/cache adaptation coverage differs")

    state = torch.load(map_path, map_location="cpu", weights_only=False)
    if (
        state.get("schema") != "lafgs_materialized_anchor_map"
        or torch.as_tensor(state.get("anchor_ids")).numel()
        != int(caches[0]["anchor_count"])
    ):
        raise ValueError("V21 Gaussian stable map registry differs from cache")
    del state

    device = torch.device(args.device)
    if gaussian_factory is None:
        from priors.models import GaussianModel2D

        gaussian_factory = GaussianModel2D
    if render_fn is None:
        from priors.rendering import render_from_pose_gsplat

        render_fn = render_from_pose_gsplat
    model = gaussian_factory(int(args.sh_degree), device=device)
    model.load_ply(gaussian_path, loc_feature_dim=0)
    model = model.to(device).eval()
    primitive_count = int(torch.as_tensor(model.get_xyz).shape[0])
    if primitive_count <= 0:
        raise ValueError("V21 Gaussian prior is empty")

    baseline_contract = caches[0]["baseline_contract"]
    pixel_center_offset = float(baseline_contract.get("pixel_center_offset", math.nan))
    if not math.isfinite(pixel_center_offset):
        raise ValueError("V21 Gaussian cache has no pixel-centre contract")
    render_contract = {
        "renderer": "priors.rendering.render_from_pose_gsplat",
        "gaussian_model": "priors.models.GaussianModel2D",
        "gaussian_type": "2dgs",
        "sh_degree": int(args.sh_degree),
        "loc_feature_dim": 0,
        "render_mode": "RGB+ED",
        "rgb_only": True,
        "requested_rasterize_mode": "antialiased",
        "effective_rasterize_mode": "omitted_unsupported_by_2dgs_wrapper",
        "rasterize_mode_argument_forwarded": False,
        "background_rgb": [0.0, 0.0, 0.0],
        "pose_source": "adaptation_test_ground_truth_w2c_delayed_feedback",
        "intrinsics_source": "native_test_camera_intrinsics",
        "fov_conversion": "2_atan(image_extent/(2*focal_length))",
        "pixel_center_offset": pixel_center_offset,
        "stored_rasters": False,
        "stored_values": "sampled_keypoint_depth_alpha_and_3x3_depth_stability_only",
    }
    cache_sources = [_source(path) for path in args.frontend_cache]
    producer_sources = [_source(REPOSITORY_ROOT / path) for path in PRODUCER_SOURCES]
    all_sources = [
        split_source,
        map_source,
        gaussian_source,
        *cache_sources,
        *producer_sources,
    ]
    records = []
    for completed, item in enumerate(ordered, start=1):
        record = item["record"]
        sampled = _render_support(
            model=model,
            record=record,
            render_fn=render_fn,
            device=device,
            render_contract=render_contract,
        )
        records.append(
            build_support_record(
                frontend_record=record,
                frontend_cache_path=item["source"]["path"],
                frontend_cache_sha256=item["source"]["sha256"],
                frontend_shard_index=item["shard_index"],
                sampled=sampled,
                pixel_center_offset=pixel_center_offset,
            )
        )
        print(
            f"V21 Gaussian support: {completed}/{len(ordered)} adaptation queries",
            flush=True,
        )

    _verify_sources(all_sources)
    source_frontend_shards = []
    for cache, source in sorted(
        zip(caches, cache_sources), key=lambda value: int(value[0]["shard_index"])
    ):
        source_frontend_shards.append(
            {
                **source,
                "shard_index": int(cache["shard_index"]),
                "shard_count": int(cache["shard_count"]),
                "query_count": int(cache["query_count"]),
            }
        )
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": ROLE,
        "training_consumers_allowed": True,
        "ground_truth_pose_is_delayed_feedback_authority": True,
        "control_or_confirmation_forbidden": True,
        "correspondence_truth_claimed": False,
        "negative_labels_created": False,
        "deployment_authority": False,
        "evidence_semantics": dict(EVIDENCE_SEMANTICS),
        "split_manifest_sha256": split_source["sha256"],
        "stable_map_sha256": map_source["sha256"],
        "gaussian_ply_sha256": gaussian_source["sha256"],
        "gaussian_primitive_count": primitive_count,
        "frontend_shard_registry": registry,
        "frontend_shard_registry_sha256": registry["registry_sha256"],
        "source_frontend_shards": source_frontend_shards,
        "query_count": len(records),
        "render_contract": render_contract,
        "render_contract_sha256": sha256_json(render_contract),
        "inputs": {
            "split_manifest": split_source,
            "stable_map": map_source,
            "gaussian_ply": gaussian_source,
            "frontend_caches": cache_sources,
            "producer_sources": producer_sources,
        },
        "records": records,
    }
    atomic_torch_save_fresh(payload, output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--frontend-cache",
        type=Path,
        action="append",
        required=True,
        help="repeat once for every adaptation frontend cache shard",
    )
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    materialize(parse_args())


if __name__ == "__main__":
    main()
