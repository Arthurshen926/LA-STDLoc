"""Audit mapping evidence density and materialize a paired K_deploy manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import load_mainline_config
from evidence.mapping_density_audit import audit_mapping_density


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--mapping-graph", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/paper_mainline.yaml")
    )
    parser.add_argument("--mapping-keypoints", type=int, default=2048)
    parser.add_argument("--expected-nms-radius", type=int, default=4)
    parser.add_argument("--deployment-keypoints", default="1024,2048")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-manifest-output", type=Path)
    args = parser.parse_args()

    config = load_mainline_config(args.config)
    factors = tuple(int(value) for value in args.deployment_keypoints.split(","))
    report = audit_mapping_density(
        scene=args.scene,
        query_cache_path=args.query_cache,
        mapping_graph_path=args.mapping_graph,
        deployment=config.values["deployment"],
        mapping_keypoints_target=args.mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        deployment_keypoint_factors=factors,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    factor_output = args.factor_manifest_output
    if factor_output is None:
        factor_output = output.with_name(output.stem + ".paired_factor_manifest.json")
    factor_output = factor_output.expanduser().resolve()
    factor_output.parent.mkdir(parents=True, exist_ok=True)
    factor_output.write_text(
        json.dumps(report["factor_manifest"], indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output": str(output),
        "factor_manifest": str(factor_output),
        "decision": report["decision"],
        "mechanism_gap_count": len(report["mechanism_gaps"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
