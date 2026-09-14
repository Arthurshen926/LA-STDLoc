"""Compare fresh prior runs, refusing reused or mismatched build protocols.

This checks commands and reports, not renderer correctness or full camera
tensor parity. It does not select a winning method or modify either run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalized_commands(manifest: dict, root: Path) -> list:
    prior = manifest["prior"]["ply"]
    prefix = str(root.resolve()) + "/"

    def normalize(value: str) -> str:
        if value == prior:
            return "<PRIOR_PLY>"
        if value.startswith(prefix):
            return "<RUN>/" + value[len(prefix):]
        return value

    return [(s["name"], [normalize(v) for v in s["command"]])
            for s in manifest["stages"]]


def check_manifest_pair(left: dict, right: dict, roots: tuple[Path, Path]) -> None:
    for manifest in (left, right):
        if manifest.get("schema") != "anygsloc_scene_experiment" or manifest.get("complete") is not True:
            raise ValueError("both scene runs must be complete")
        names = [stage["name"] for stage in manifest["stages"]]
        if len(names) != len(set(names)) or not {"observations", "projective_map", "base_evaluation"} <= set(names):
            raise ValueError("fresh end-to-end stages are missing or duplicated")
        if not any(name.startswith("v2_audit_") for name in names):
            raise ValueError("V2 audit stages are missing")
        if any(manifest["results"].get(name, {}).get("status") != "executed" for name in names):
            raise ValueError("protocol comparison requires fresh stages, not reused artifacts")
        scope = manifest["scientific_scope"]
        if not scope or any(value is not False for value in scope.values()):
            raise ValueError("protocol is outside feedback-free, source-image-free scope")
    for field in ("scene", "dataset", "config", "scientific_scope", "mapping_detection"):
        if left[field] != right[field]:
            raise ValueError(f"run protocol differs: {field}")
    for field in ("gaussian_type", "sh_degree", "white_background"):
        if left["prior"][field] != right["prior"][field]:
            raise ValueError(f"render protocol differs: {field}")
    if normalized_commands(left, roots[0]) != normalized_commands(right, roots[1]):
        raise ValueError("stage commands differ beyond prior PLY and output root")


def compare(roots: tuple[Path, Path]) -> dict:
    def read(root: Path, relative: str) -> dict:
        return json.loads((root / relative).read_text())

    manifests = [read(root, "anygsloc_scene_manifest.json") for root in roots]
    check_manifest_pair(*manifests, roots)
    maps = [read(root, "projective_map/report.json") for root in roots]
    # Completion availability is an outcome, not a policy parameter.
    outcome_fields = {"projective_completion_available", "projective_completion_unavailable_reason"}
    contracts = [{k: v for k, v in m["contracts"].items() if k not in outcome_fields}
                 for m in maps]
    if not contracts[0] or contracts[0] != contracts[1]:
        raise ValueError("map construction contracts differ")
    import torch
    from common.hashing import sha256_file

    identities = []
    mapping_names = []
    for root, report in zip(roots, maps):
        path = root / "projective_map/projective_anchor_map.pt"
        if sha256_file(path) != report["output"]["map_sha256"]:
            raise ValueError("map bytes differ from construction report")
        state = torch.load(path, map_location="cpu", weights_only=False)
        identities.append(state.get("provenance", {}).get("implementation_sha256"))
        mapping_names.append(state.get("v6_mapping_query_names"))
        del state
    if not identities[0] or identities[0] != identities[1]:
        raise ValueError("map implementation hashes differ or are missing")
    if not mapping_names[0] or mapping_names[0] != mapping_names[1]:
        raise ValueError("ordered mapping registries differ or are missing")
    if maps[0]["counts"]["mapping_views"] != maps[1]["counts"]["mapping_views"]:
        raise ValueError("mapping view counts differ")
    summaries = []
    registries = []
    for root, manifest in zip(roots, manifests):
        stage = next(s for s in manifest["stages"] if s["name"] == "base_evaluation")
        summary_path = Path(stage["output"])
        summaries.append(json.loads(summary_path.read_text()))
        rows = json.loads((summary_path.parent / "results.json").read_text())
        names = [row["image_name"] for row in rows]
        if len(names) != len(set(names)) or len(names) != summaries[-1]["query_count"]:
            raise ValueError("evaluation query registry is incomplete or duplicated")
        registries.append(names)
    if registries[0] != registries[1]:
        raise ValueError("ordered evaluation query registries differ")
    for field in ("evaluated_split", "random_seed", "ransac_confidence",
                  "ransac_minimum_iterations", "ransac_maximum_iterations"):
        if field not in summaries[0] or summaries[0][field] != summaries[1].get(field):
            raise ValueError(f"evaluation protocol differs: {field}")
    metrics = ("median_te_cm", "mean_te_cm", "p90_te_cm", "median_ae_deg",
               "mean_ae_deg", "p90_ae_deg", "catastrophic_100cm_count",
               "matching_ms_mean", "ransac_ms_mean", "total_ms_mean")
    return {
        "status": "COMMAND_AND_REPORT_PARITY_PASS",
        "full_camera_tensor_parity_checked": False,
        "independent_generalization_evidence": False,
        "timing_caveat": "Shared-device timings are not isolated algorithm benchmarks.",
        "runs": [{"root": str(root.resolve()), "counts": mapping["counts"],
                  "metrics": {key: summary[key] for key in metrics}}
                 for root, mapping, summary in zip(roots, maps, summaries)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare((args.left.resolve(), args.right.resolve())), indent=2))


if __name__ == "__main__":
    main()
