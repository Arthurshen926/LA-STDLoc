#!/usr/bin/env python3
"""Observe and apply reversible map-side feedback on the V2 rebuilt map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v7_feedback import (
    DiagnosticRegistry,
    diagnose_feedback_query,
    load_v7_fixed_plant,
    localize_rgb_query,
)
from map_learning.v7_descriptor_controller import reconstruct_v7_descriptors
from map_learning.v8_feedback_controller import (
    materialize_quarantined_map,
    propose_feedback_anchor_quarantine,
)
from topology.v6_anchor_map import identity_metric_state


def _save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _require_sha(path: Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{label} SHA256 differs")
    return actual


def _remap_certificate_rows(
    certificate: dict,
    source_keypoints: torch.Tensor,
    replay_keypoints: torch.Tensor,
) -> tuple[dict, float]:
    """Bind persisted V2 row evidence to a numerically replayed Top-K list.

    GPU-dependent score ties may alter a small Top-K tail.  Only exact integer
    coordinate matches inherit the old V2 decision; unmatched replay rows are
    fail-closed invalid rather than receiving a positional mask by mistake.
    """

    source = torch.as_tensor(source_keypoints).long().cpu()
    replay = torch.as_tensor(replay_keypoints).long().cpu()
    valid = torch.as_tensor(certificate["row_valid"]).bool().reshape(-1).cpu()
    if valid.shape != source.shape[:1] or source.shape[1:] != (2,):
        raise ValueError("persisted certificate rows do not align with keypoints")
    coordinate_to_row = {tuple(xy): row for row, xy in enumerate(source.tolist())}
    output_valid = torch.zeros(replay.shape[0], dtype=torch.bool)
    matched = 0
    for row, xy in enumerate(replay.tolist()):
        source_row = coordinate_to_row.get(tuple(xy))
        if source_row is not None:
            output_valid[row] = valid[source_row]
            matched += 1
    output = dict(certificate)
    output["row_valid"] = output_valid
    output["row_evidence_remapped_by_exact_keypoint_coordinate"] = True
    output["unmatched_replay_rows_fail_closed"] = int(replay.shape[0] - matched)
    return output, matched / max(int(replay.shape[0]), 1)


@torch.inference_mode()
def observe(args: argparse.Namespace) -> dict:
    batch_path = args.certified_batch.resolve()
    map_path = args.map.resolve()
    metric_path = args.metric.resolve()
    batch_sha = _require_sha(batch_path, args.expected_certified_batch_sha256, "batch")
    map_sha = _require_sha(map_path, args.expected_map_sha256, "map")
    metric_sha = _require_sha(metric_path, args.expected_metric_sha256, "metric")
    batch = json.loads(batch_path.read_text())
    if not (
        batch.get("view_role") == "feedback_query"
        and batch.get("uses_test_queries") is False
        and batch.get("map_mutation_count") == 0
    ):
        raise ValueError("V8 observation requires an immutable non-test feedback batch")
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    provenance = state.get("provenance", {})
    construction = state.get("projective_anchor_construction", {})
    if not (
        provenance.get("mapping_source")
        == "gaussian_render_v2_filtered_before_projective_association"
        and construction.get("v2_preassociation_filter") is True
    ):
        raise ValueError("V8 feedback M0 is not the V2 pre-association rebuild")
    ids = torch.as_tensor(state["anchor_ids"]).long().cpu()
    registry = DiagnosticRegistry(
        anchor_ids=ids,
        anchor_xyz=torch.as_tensor(state["anchor_xyz"]).float().cpu(),
        eligible=torch.ones(ids.numel(), dtype=torch.bool),
    )
    plant = load_v7_fixed_plant(
        map_path, metric_path, device=args.device, diagnostic_registry=registry
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir()
    counts = {
        "representation_deficit": 0,
        "precision_deficit": 0,
        "coverage_deficit": 0,
        "unreliable_query": 0,
        "nominal_success": 0,
    }
    output_records = []
    for index, item in enumerate(batch["records"]):
        source_path = Path(item["path"]).resolve()
        source_sha = _require_sha(source_path, item["sha256"], "feedback record")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        localization = localize_rgb_query(
            source["rgb_float16"].float(), source["intrinsics"], plant
        )
        certificate, replay_fraction = _remap_certificate_rows(
            source["certificate"], source["keypoints"], localization.keypoints
        )
        if replay_fraction < 0.98:
            raise RuntimeError("V8 feedback frontend replay overlap is below 98%")
        diagnosis = diagnose_feedback_query(
            localization,
            source["pose_w2c"],
            source["alpha_float16"],
            source["depth_float16"],
            certificate,
        )
        counts[diagnosis["category"]] += 1
        record = {
            "schema": "lafgs_v8_map_feedback_record",
            "version": 1,
            "query_index": int(source["query_index"]),
            "pose_family_id": int(source["pose_family_id"]),
            "source_record": str(source_path),
            "source_record_sha256": source_sha,
            "frontend_exact_keypoint_replay_fraction": replay_fraction,
            "unmatched_replay_rows_fail_closed": certificate[
                "unmatched_replay_rows_fail_closed"
            ],
            "diagnosis": diagnosis,
            "estimated_pose_w2c": torch.from_numpy(localization.pose.pose_w2c),
            "matches": {
                "keypoint_indices": localization.matches.keypoint_indices,
                "anchor_indices": localization.matches.anchor_indices,
                "scores": localization.matches.scores,
            },
            "map_mutation_count": 0,
            "uses_test_queries": False,
        }
        path = records_dir / f"query_{int(source['query_index']):04d}.pt"
        _save(record, path)
        output_records.append(
            {
                "query_index": record["query_index"],
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "category": diagnosis["category"],
                "can_drive_map_update": bool(diagnosis["can_drive_map_update"]),
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == len(batch["records"]):
            print(f"feedback observe {index + 1}/{len(batch['records'])}", flush=True)
    manifest = {
        "schema": "lafgs_v8_map_feedback_batch",
        "version": 1,
        "phase": "observe",
        "status": "PASS",
        "view_role": "feedback_query",
        "query_count": len(output_records),
        "category_counts": counts,
        "update_authorized_count": sum(
            int(item["can_drive_map_update"]) for item in output_records
        ),
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "map_mutation_count": 0,
        "query_detector_used": False,
        "input": {
            "certified_batch": str(batch_path),
            "certified_batch_sha256": batch_sha,
            "map": str(map_path),
            "map_sha256": map_sha,
            "metric": str(metric_path),
            "metric_sha256": metric_sha,
        },
        "records": output_records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def propose(args: argparse.Namespace) -> dict:
    feedback_path = args.feedback_batch.resolve()
    map_path = args.map.resolve()
    feedback_sha = _require_sha(
        feedback_path, args.expected_feedback_batch_sha256, "feedback"
    )
    map_sha = _require_sha(map_path, args.expected_map_sha256, "map")
    feedback = json.loads(feedback_path.read_text())
    if not (
        feedback.get("schema") == "lafgs_v8_map_feedback_batch"
        and feedback.get("uses_test_queries") is False
        and feedback.get("map_mutation_count") == 0
        and feedback.get("input", {}).get("map_sha256") == map_sha
    ):
        raise ValueError("quarantine proposal is not bound to immutable V8 M0")
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    records = []
    for item in feedback["records"]:
        path = Path(item["path"]).resolve()
        _require_sha(path, item["sha256"], "feedback observation")
        records.append(torch.load(path, map_location="cpu", weights_only=False))
    proposal = propose_feedback_anchor_quarantine(
        anchor_ids=state["anchor_ids"],
        feedback_records=records,
        minimum_pose_families=args.minimum_pose_families,
        minimum_queries=args.minimum_queries,
        minimum_query_task_gain=args.minimum_query_task_gain,
        maximum_quarantine_fraction=args.maximum_quarantine_fraction,
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    proposal_path = args.output_dir / "quarantine_proposal.pt"
    _save(proposal, proposal_path)
    output_map = None
    output_metric = None
    if proposal["proposed_anchor_count"]:
        candidate, _ = materialize_quarantined_map(
            state, proposal["proposed_anchor_rows"]
        )
        candidate["provenance"] = {
            **dict(candidate["provenance"]),
            "v8_feedback_batch": str(feedback_path),
            "v8_feedback_batch_sha256": feedback_sha,
        }
        output_map = args.output_dir / "projective_anchor_map.pt"
        _save(candidate, output_map)
        output_metric = args.output_dir / "identity_metric.pt"
        metric = identity_metric_state(
            candidate,
            map_path=str(output_map.resolve()),
            map_sha256=sha256_file(output_map),
        )
        _save(metric, output_metric)
    report = {
        "schema": "lafgs_v8_map_feedback_quarantine_report",
        "version": 1,
        "phase": "propose",
        "status": "PASS",
        "uses_test_queries": False,
        "query_detector_used": False,
        "feedback_descriptors_copied": False,
        "input": {
            "feedback_batch": str(feedback_path),
            "feedback_batch_sha256": feedback_sha,
            "map": str(map_path),
            "map_sha256": map_sha,
        },
        "proposal": {
            key: value
            for key, value in proposal.items()
            if not isinstance(value, torch.Tensor) and key != "candidate_audit"
        },
        "output": {
            "proposal": str(proposal_path.resolve()),
            "proposal_sha256": sha256_file(proposal_path),
            "map": None if output_map is None else str(output_map.resolve()),
            "map_sha256": None if output_map is None else sha256_file(output_map),
            "metric": None if output_metric is None else str(output_metric.resolve()),
            "metric_sha256": (
                None if output_metric is None else sha256_file(output_metric)
            ),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def descriptor(args: argparse.Namespace) -> dict:
    """Build a descriptor-only proposal from original V2-valid observations."""

    feedback_path = args.feedback_batch.resolve()
    map_path = args.map.resolve()
    candidate_path = args.candidate_pool.resolve()
    cache_path = args.mapping_feature_cache.resolve()
    feedback_sha = _require_sha(
        feedback_path, args.expected_feedback_batch_sha256, "feedback"
    )
    map_sha = _require_sha(map_path, args.expected_map_sha256, "map")
    candidate_sha = _require_sha(
        candidate_path, args.expected_candidate_pool_sha256, "candidates"
    )
    cache_sha = _require_sha(
        cache_path, args.expected_mapping_feature_cache_sha256, "mapping cache"
    )
    feedback = json.loads(feedback_path.read_text())
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    candidates = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if not torch.equal(
        torch.as_tensor(state["anchor_ids"]).long(),
        torch.as_tensor(candidates["anchor_ids"]).long(),
    ):
        raise ValueError("V8 descriptor candidate registry differs from M0")
    evidence_rows = []
    signs: dict[int, dict[str, set[int]]] = {}
    for item in feedback["records"]:
        path = Path(item["path"]).resolve()
        _require_sha(path, item["sha256"], "feedback observation")
        record = torch.load(path, map_location="cpu", weights_only=False)
        diagnosis = record["diagnosis"]
        if diagnosis.get("can_drive_map_update") is not True:
            continue
        control = diagnosis["descriptor_control_evidence"]
        row = {
            "pose_family_id": int(record["pose_family_id"]),
            "query_descriptors": control["query_descriptors"],
            "positive_anchor_ids": control["positive_anchor_ids"],
            "false_attractor_anchor_ids": control["false_attractor_anchor_ids"],
        }
        evidence_rows.append(row)
        family = row["pose_family_id"]
        for anchor_id in torch.as_tensor(row["positive_anchor_ids"]).tolist():
            signs.setdefault(int(anchor_id), {"positive": set(), "harm": set()})[
                "positive"
            ].add(family)
        for anchor_id in torch.as_tensor(
            row["false_attractor_anchor_ids"]
        ).tolist():
            signs.setdefault(int(anchor_id), {"positive": set(), "harm": set()})[
                "harm"
            ].add(family)
    potential = {
        anchor_id
        for anchor_id, kinds in signs.items()
        if len(kinds["positive"] | kinds["harm"]) >= args.minimum_pose_families
        and not bool(kinds["positive"] & kinds["harm"])
    }
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not (
        cache.get("uses_source_mapping_rgb") is False
        and cache.get("uses_test_queries") is False
    ):
        raise ValueError("descriptor reconstruction requires mapping renders only")
    id_to_row = {
        int(anchor_id): row
        for row, anchor_id in enumerate(candidates["anchor_ids"].tolist())
    }
    csr = candidates["projective_anchor_observations"]
    offsets = torch.as_tensor(csr["observation_offsets"]).long()
    queries = torch.as_tensor(csr["query_indices"]).long()
    keypoints = torch.as_tensor(csr["keypoint_indices"]).long()
    names = candidates["query_names"]
    bins = torch.as_tensor(candidates["query_bins"]).long()
    banks = {}
    for anchor_id in potential:
        anchor_row = id_to_row[anchor_id]
        start, stop = int(offsets[anchor_row]), int(offsets[anchor_row + 1])
        descriptors, families = [], []
        for query, keypoint in zip(
            queries[start:stop].tolist(), keypoints[start:stop].tolist()
        ):
            descriptors.append(
                torch.as_tensor(
                    cache["queries"][names[query]]["native_descriptors"][keypoint]
                ).float()
            )
            families.append(int(bins[query]))
        banks[anchor_id] = {
            "descriptors": torch.stack(descriptors),
            "view_families": torch.tensor(families, dtype=torch.long),
        }
    reconstruction = reconstruct_v7_descriptors(
        anchor_ids=state["anchor_ids"],
        current_descriptors=state["anchor_features"],
        feedback_evidence=evidence_rows,
        observation_banks=banks,
        minimum_pose_families=args.minimum_pose_families,
        learning_rate=args.descriptor_learning_rate,
        harmful_weight=args.descriptor_harmful_weight,
        maximum_descriptor_angle_deg=args.maximum_descriptor_angle_deg,
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    reconstruction_path = args.output_dir / "descriptor_reconstruction.pt"
    _save(reconstruction, reconstruction_path)
    output_map = None
    output_metric = None
    if reconstruction["changed_anchor_count"]:
        output = dict(state)
        output["anchor_features"] = reconstruction["anchor_features"]
        output["provenance"] = {
            **dict(state.get("provenance", {})),
            "v8_feedback_descriptor_reconstruction": True,
            "v8_feedback_batch": str(feedback_path),
            "v8_feedback_batch_sha256": feedback_sha,
            "feedback_descriptors_copied_into_map": False,
            "uses_test_queries": False,
        }
        output_map = args.output_dir / "projective_anchor_map.pt"
        _save(output, output_map)
        output_metric = args.output_dir / "identity_metric.pt"
        metric = identity_metric_state(
            output,
            map_path=str(output_map.resolve()),
            map_sha256=sha256_file(output_map),
        )
        _save(metric, output_metric)
    report = {
        "schema": "lafgs_v8_map_feedback_descriptor_report",
        "version": 1,
        "phase": "descriptor",
        "status": "PASS",
        "uses_test_queries": False,
        "query_detector_used": False,
        "feedback_descriptors_copied": False,
        "evidence_query_count": len(evidence_rows),
        "potential_anchor_count": len(potential),
        "changed_anchor_count": int(reconstruction["changed_anchor_count"]),
        "input": {
            "feedback_batch": str(feedback_path),
            "feedback_batch_sha256": feedback_sha,
            "map": str(map_path),
            "map_sha256": map_sha,
            "candidate_pool": str(candidate_path),
            "candidate_pool_sha256": candidate_sha,
            "mapping_feature_cache": str(cache_path),
            "mapping_feature_cache_sha256": cache_sha,
        },
        "output": {
            "reconstruction": str(reconstruction_path.resolve()),
            "reconstruction_sha256": sha256_file(reconstruction_path),
            "map": None if output_map is None else str(output_map.resolve()),
            "map_sha256": None if output_map is None else sha256_file(output_map),
            "metric": None if output_metric is None else str(output_metric.resolve()),
            "metric_sha256": (
                None if output_metric is None else sha256_file(output_metric)
            ),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("observe", "propose", "descriptor"), required=True
    )
    parser.add_argument("--certified-batch", type=Path)
    parser.add_argument("--feedback-batch", type=Path)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric", type=Path)
    parser.add_argument("--candidate-pool", type=Path)
    parser.add_argument("--mapping-feature-cache", type=Path)
    parser.add_argument("--expected-certified-batch-sha256")
    parser.add_argument("--expected-feedback-batch-sha256")
    parser.add_argument("--expected-map-sha256")
    parser.add_argument("--expected-metric-sha256")
    parser.add_argument("--expected-candidate-pool-sha256")
    parser.add_argument("--expected-mapping-feature-cache-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-pose-families", type=int, default=2)
    parser.add_argument("--minimum-queries", type=int, default=2)
    parser.add_argument("--minimum-query-task-gain", type=float, default=0.01)
    parser.add_argument("--maximum-quarantine-fraction", type=float, default=0.01)
    parser.add_argument("--descriptor-learning-rate", type=float, default=0.35)
    parser.add_argument("--descriptor-harmful-weight", type=float, default=1.0)
    parser.add_argument("--maximum-descriptor-angle-deg", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "observe" and (args.certified_batch is None or args.metric is None):
        parser.error("observe requires --certified-batch and --metric")
    if args.phase == "propose" and args.feedback_batch is None:
        parser.error("propose requires --feedback-batch")
    if args.phase == "descriptor" and (
        args.feedback_batch is None
        or args.candidate_pool is None
        or args.mapping_feature_cache is None
    ):
        parser.error(
            "descriptor requires --feedback-batch, --candidate-pool and "
            "--mapping-feature-cache"
        )
    runners = {"observe": observe, "propose": propose, "descriptor": descriptor}
    report = runners[args.phase](args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
