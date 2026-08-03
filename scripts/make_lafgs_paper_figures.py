#!/usr/bin/env python3
"""Generate frozen LaFGS paper figures from formal artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lafgs.protocol import load_mainline_protocol
from lafgs.visualization.paper import (
    build_anysplat_a0_a1_figure,
    build_method_overview,
    build_topology_distillation_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lafgs_paper_mainline.yaml")
    parser.add_argument("--lafgs-root", required=True)
    parser.add_argument("--prior-quality", required=True)
    parser.add_argument("--render-path", required=True)
    parser.add_argument("--query-camera-source", required=True)
    parser.add_argument("--query-image-root", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--query-name", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_mainline_protocol(args.config)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.lafgs_root).expanduser().resolve()
    metric_steps = int(protocol.resolved["reconstruction"]["metric_steps"])
    overview = build_method_overview(protocol, output / "figure_method_overview.png")
    topology = build_topology_distillation_figure(
        a0_map_path=root / "runs/frozen_v1/bootstrap/0_lafgs_map_state.pt",
        a1_map_path=(
            root
            / "self_localization_reconstruction"
            / f"anchor_map_step_{metric_steps:04d}.pt"
        ),
        output_path=output / "figure_topology_distillation.png",
        seed=args.seed,
    )
    comparison = build_anysplat_a0_a1_figure(
        protocol=protocol,
        lafgs_root=root,
        prior_quality_path=args.prior_quality,
        render_path=args.render_path,
        query_camera_source=args.query_camera_source,
        query_image_root=args.query_image_root,
        mask_path=args.mask_path,
        output_dir=output,
        seed=args.seed,
        query_name=args.query_name,
        device=args.device,
    )
    report = {
        "schema": "lafgs_paper_figure_set",
        "version": 1,
        "protocol": protocol.manifest(),
        "figures": {"overview": overview, "topology": topology, "A0_A1": comparison},
    }
    report_path = output / "figure_set_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
