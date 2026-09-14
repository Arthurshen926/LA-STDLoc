"""Release commands; no dataset evaluator or renderer is imported for localization."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "build-map":
        from anygsloc.mapping.build import main as build

        return build(argv[1:])
    if argv and argv[0] == "evaluate":
        from anygsloc.evaluation.runner import main as evaluate

        return evaluate(argv[1:])
    p = argparse.ArgumentParser(prog="anygsloc")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build-map", help="frozen offline build; --dry-run writes nothing")
    sub.add_parser(
        "evaluate", help="frozen dataset evaluation; never launched by build-map"
    )
    e = sub.add_parser("export-map")
    e.add_argument("--source", required=True)
    e.add_argument("--metric", required=True)
    e.add_argument("--output", required=True)
    c = sub.add_parser(
        "compact-map", help="mapping-only compact selection; never launches evaluation"
    )
    c.add_argument("--map-dir", required=True)
    c.add_argument("--cache", required=True)
    c.add_argument("--output", required=True)
    localize = sub.add_parser("localize")
    localize.add_argument("--map", required=True)
    localize.add_argument(
        "--query",
        required=True,
        help="trusted .pt containing RGB tensor, fov_x/fov_y, optional valid_mask",
    )
    localize.add_argument("--keypoints", type=int, required=True)
    localize.add_argument("--reprojection-px", type=float, required=True)
    localize.add_argument(
        "--mask-policy", choices=["none", "explicit_valid_mask"], required=True
    )
    localize.add_argument("--device", default="cuda")
    localize.add_argument("--seed", type=int, default=2026)
    localize.add_argument("--output", required=True)
    a = p.parse_args(argv)
    if a.command == "compact-map":
        from anygsloc.mapping.compact import compact_map

        print(json.dumps(compact_map(a.map_dir, a.cache, a.output), indent=2))
        return
    if a.command == "export-map":
        from anygsloc.artifacts import export_map

        print(json.dumps(export_map(a.source, a.metric, a.output), indent=2))
        return
    import torch
    from anygsloc.online.runtime import Localizer
    from anygsloc.artifacts import sha256
    from anygsloc.config import FROZEN

    torch.set_num_threads(FROZEN.cpu_threads)

    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    q = torch.load(a.query, map_location="cpu", weights_only=True)
    engine = Localizer(
        a.map,
        keypoint_count=a.keypoints,
        reprojection_error_px=a.reprojection_px,
        mask_policy=a.mask_policy,
        device=a.device,
        seed=a.seed,
    )
    r = engine.localize(
        q["image"], fov_x=q["fov_x"], fov_y=q["fov_y"], valid_mask=q.get("valid_mask")
    )
    payload = dict(
        pose_w2c=r["pose"].pose_w2c.tolist(),
        inliers=r["pose"].inliers.tolist(),
        success=r["success"],
        resolved_config=r["resolved_config"],
        seed=a.seed,
        map_sha256=sha256(a.map),
        query_sha256=sha256(a.query),
        runtime_ms=r["runtime_ms"],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
