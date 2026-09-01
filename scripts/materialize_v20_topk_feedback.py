#!/usr/bin/env python3
"""Materialize leakage-safe V20 Top-K competition evidence from V19 truth."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v20_feedback import (
    build_topk_competition_evidence,
    partition_feedback_rows,
)
from map_learning.v18_provenance_truth import TRUTH_EQUIVALENT, TRUTH_UNIQUE


def _save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _subset_truth(truth: dict, rows: torch.Tensor) -> dict:
    rows = torch.as_tensor(rows).long().reshape(-1)
    offsets = torch.as_tensor(truth["truth_offsets"]).long()
    anchors = torch.as_tensor(truth["truth_anchor_rows"]).long()
    row_count = int(truth["row_count"])
    if (
        offsets.shape != (row_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != anchors.numel()
        or bool(((offsets[1:] - offsets[:-1]) < 0).any())
    ):
        raise ValueError("V20 teacher truth CSR is invalid")
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= row_count):
        raise ValueError("V20 truth subset row is outside the teacher registry")
    parts = [anchors[int(offsets[row]) : int(offsets[row + 1])] for row in rows]
    counts = torch.tensor([part.numel() for part in parts], dtype=torch.long)
    return {
        **truth,
        "row_count": int(rows.numel()),
        "truth_status": torch.as_tensor(truth["truth_status"])[rows].clone(),
        "truth_offsets": torch.cat(
            (torch.zeros(1, dtype=torch.long), counts.cumsum(0))
        ),
        "truth_anchor_rows": (
            torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
        ),
    }


def _csr(parts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.tensor([part.numel() for part in parts], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    values = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
    return offsets, values


def _truth_anchor_rows(truth: dict, row: int, anchor_count: int) -> torch.Tensor:
    offsets = torch.as_tensor(truth["truth_offsets"]).long().reshape(-1)
    anchors = torch.as_tensor(truth["truth_anchor_rows"]).long().reshape(-1)
    row_count = int(truth["row_count"])
    if (
        offsets.shape != (row_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != anchors.numel()
        or bool(((offsets[1:] - offsets[:-1]) < 0).any())
    ):
        raise ValueError("V20 teacher truth CSR is invalid")
    if not 0 <= int(row) < row_count:
        raise ValueError("V20 teacher truth row is outside the registry")
    values = anchors[int(offsets[row]) : int(offsets[row + 1])]
    if values.numel() and (
        int(values.min()) < 0 or int(values.max()) >= int(anchor_count)
    ):
        raise ValueError("V20 teacher truth references an invalid Anchor")
    return values


def _bind_v9_training_rows(
    *,
    training: dict,
    candidate_anchor_rows: torch.Tensor,
    candidate_scores: torch.Tensor,
    truth: dict,
    equivalence_class_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Authorize only V9 wrong→right rows that V19 independently confirms."""

    candidates = torch.as_tensor(candidate_anchor_rows).long()
    scores = torch.as_tensor(candidate_scores).float()
    equivalence = torch.as_tensor(equivalence_class_ids).long().reshape(-1)
    if (
        candidates.ndim != 2
        or candidates.shape != scores.shape
        or candidates.shape[1] < 2
        or int(truth["row_count"]) != candidates.shape[0]
    ):
        raise ValueError("V20 V9/V19 training registries do not align")
    if candidates.numel() and (
        int(candidates.min()) < 0 or int(candidates.max()) >= equivalence.numel()
    ):
        raise ValueError("V20 observer Top-K references an invalid Anchor")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("V20 observer Top-K scores must be finite")
    if bool((scores[:, 0] + 1e-6 < scores.max(1).values).any()):
        raise ValueError("V20 observer Top-K column zero is not the current winner")

    local_rows = torch.as_tensor(training["query_rows"]).long().reshape(-1)
    positives = torch.as_tensor(training["positive_anchor_rows"]).long().reshape(-1)
    negatives = torch.as_tensor(training["negative_anchor_rows"]).long().reshape(-1)
    ranks = torch.as_tensor(training["positive_rank"]).long().reshape(-1)
    descriptors = torch.as_tensor(training["query_descriptors"])
    pose_entered = torch.as_tensor(
        training.get("alternative_pose_entered_mask", ())
    ).bool().reshape(-1)
    row_count = local_rows.numel()
    if not (
        positives.numel()
        == negatives.numel()
        == ranks.numel()
        == descriptors.shape[0]
        == pose_entered.numel()
        == row_count
    ):
        raise ValueError("V20 V9 training evidence columns do not align")
    if not bool(pose_entered.all()):
        raise ValueError("V20 V9 training row did not enter the causal pose replay")
    if local_rows.numel() and (
        int(local_rows.min()) < 0
        or int(local_rows.max()) >= candidates.shape[0]
        or torch.unique(local_rows).numel() != local_rows.numel()
    ):
        raise ValueError("V20 V9 training query rows are invalid or duplicated")
    if ranks.numel() and (
        int(ranks.min()) < 0 or int(ranks.max()) >= candidates.shape[1]
    ):
        raise ValueError("V20 V9 positive rank is outside Top-K")
    if positives.numel() and (
        int(positives.min()) < 0
        or int(positives.max()) >= equivalence.numel()
        or int(negatives.min()) < 0
        or int(negatives.max()) >= equivalence.numel()
    ):
        raise ValueError("V20 V9 training pair references an invalid Anchor")

    status = torch.as_tensor(truth["truth_status"]).long().reshape(-1)
    if status.numel() != candidates.shape[0]:
        raise ValueError("V20 teacher truth status does not align with Top-K")
    keep = torch.ones(row_count, dtype=torch.bool)
    reasons: Counter[str] = Counter()
    for index, row in enumerate(local_rows.tolist()):
        positive = int(positives[index])
        negative = int(negatives[index])
        rank = int(ranks[index])
        row_reasons = []
        if positive != int(candidates[row, rank]):
            row_reasons.append("positive_not_bound_to_v9_rank")
        if negative != int(candidates[row, 0]):
            row_reasons.append("negative_not_current_winner")
        if int(status[row]) not in {TRUTH_UNIQUE, TRUTH_EQUIVALENT}:
            row_reasons.append("v19_truth_not_decisive")
        truth_rows = _truth_anchor_rows(truth, row, equivalence.numel())
        truth_classes = set(equivalence[truth_rows].tolist())
        if int(equivalence[positive]) not in truth_classes:
            row_reasons.append("v9_positive_not_v19_positive")
        if int(equivalence[negative]) in truth_classes:
            row_reasons.append("current_winner_is_v19_positive")
        if int(equivalence[positive]) == int(equivalence[negative]):
            row_reasons.append("v9_pair_is_equivalent")
        if row_reasons:
            keep[index] = False
            reasons.update(row_reasons)
        else:
            reasons["authorized"] += 1
    reasons["input"] += int(row_count)
    reasons["rejected"] += int((~keep).sum())
    return keep, dict(reasons)


def _truth_bound_clean_protection(
    *,
    clean: dict,
    candidate_anchor_rows: torch.Tensor,
    candidate_scores: torch.Tensor,
    truth: dict,
    equivalence_class_ids: torch.Tensor,
) -> tuple[dict, dict]:
    """Bind every clean descriptor to an explicit row and V19 non-positive set."""

    descriptors = torch.as_tensor(clean["query_descriptors"]).float()
    positives = torch.as_tensor(clean["positive_anchor_rows"]).long().reshape(-1)
    legacy_negatives = torch.as_tensor(clean["negative_anchor_rows"]).long().reshape(-1)
    sample_count = descriptors.shape[0]
    empty = {
        "query_descriptors": descriptors[:0],
        "query_rows": torch.empty(0, dtype=torch.long),
        "positive_anchor_rows": [],
        "negative_anchor_rows": [],
        "initial_margin": torch.empty(0),
    }
    if sample_count == 0:
        return empty, {"input": 0, "authorized": 0, "rejected": 0}
    if "query_rows" not in clean:
        return empty, {
            "input": int(sample_count),
            "authorized": 0,
            "rejected": int(sample_count),
            "missing_explicit_query_rows": int(sample_count),
        }

    rows = torch.as_tensor(clean["query_rows"]).long().reshape(-1)
    candidates = torch.as_tensor(candidate_anchor_rows).long()
    scores = torch.as_tensor(candidate_scores).float()
    equivalence = torch.as_tensor(equivalence_class_ids).long().reshape(-1)
    if (
        descriptors.ndim != 2
        or positives.numel() != sample_count
        or legacy_negatives.numel() != sample_count
        or rows.numel() != sample_count
        or candidates.ndim != 2
        or candidates.shape != scores.shape
        or candidates.shape[1] < 2
        or int(truth["row_count"]) != candidates.shape[0]
    ):
        raise ValueError("V20 clean protection columns do not align")
    if rows.numel() and (
        int(rows.min()) < 0
        or int(rows.max()) >= candidates.shape[0]
        or torch.unique(rows).numel() != rows.numel()
    ):
        raise ValueError("V20 clean protection query rows are invalid or duplicated")
    if candidates.numel() and (
        int(candidates.min()) < 0 or int(candidates.max()) >= equivalence.numel()
    ):
        raise ValueError("V20 clean Top-K references an invalid Anchor")
    if not bool(torch.isfinite(descriptors).all()) or not bool(
        torch.isfinite(scores).all()
    ):
        raise ValueError("V20 clean descriptors and scores must be finite")
    if bool((scores[:, 0] + 1e-6 < scores.max(1).values).any()):
        raise ValueError("V20 clean Top-K column zero is not the current winner")

    status = torch.as_tensor(truth["truth_status"]).long().reshape(-1)
    if status.numel() != candidates.shape[0]:
        raise ValueError("V20 clean truth status does not align with Top-K")
    selected_descriptors = []
    selected_rows = []
    selected_positive = []
    selected_negative = []
    margins = []
    reasons: Counter[str] = Counter(input=int(sample_count))
    for index, row in enumerate(rows.tolist()):
        positive = int(positives[index])
        old_negative = int(legacy_negatives[index])
        row_reasons = []
        if positive != int(candidates[row, 0]):
            row_reasons.append("positive_not_current_winner")
        if old_negative != int(candidates[row, 1]):
            row_reasons.append("legacy_negative_not_current_top2")
        if int(status[row]) not in {TRUTH_UNIQUE, TRUTH_EQUIVALENT}:
            row_reasons.append("v19_truth_not_decisive")
        truth_rows = _truth_anchor_rows(truth, row, equivalence.numel())
        truth_classes = set(equivalence[truth_rows].tolist())
        if positive < 0 or positive >= equivalence.numel():
            raise ValueError("V20 clean positive references an invalid Anchor")
        if int(equivalence[positive]) not in truth_classes:
            row_reasons.append("current_winner_not_v19_positive")
        nonpositive = []
        seen = set()
        for value in candidates[row].tolist():
            anchor = int(value)
            if int(equivalence[anchor]) in truth_classes or anchor in seen:
                continue
            seen.add(anchor)
            nonpositive.append(anchor)
        if not nonpositive:
            row_reasons.append("no_v19_proven_nonpositive_in_topk")
        if old_negative < 0 or old_negative >= equivalence.numel():
            raise ValueError("V20 clean negative references an invalid Anchor")
        elif int(equivalence[old_negative]) in truth_classes:
            row_reasons.append("legacy_top2_is_v19_positive")
        if row_reasons:
            reasons.update(row_reasons)
            reasons["rejected"] += 1
            continue
        negative = torch.tensor(nonpositive, dtype=torch.long)
        negative_scores = torch.tensor(
            [
                float(scores[row, column])
                for column, anchor in enumerate(candidates[row].tolist())
                if int(anchor) in seen
            ]
        )
        selected_descriptors.append(descriptors[index])
        selected_rows.append(row)
        selected_positive.append(torch.tensor([positive], dtype=torch.long))
        selected_negative.append(negative)
        margins.append(float(scores[row, 0] - negative_scores.max()))
        reasons["authorized"] += 1
    return (
        {
            "query_descriptors": (
                torch.stack(selected_descriptors)
                if selected_descriptors
                else descriptors[:0]
            ),
            "query_rows": torch.tensor(selected_rows, dtype=torch.long),
            "positive_anchor_rows": selected_positive,
            "negative_anchor_rows": selected_negative,
            "initial_margin": torch.tensor(margins, dtype=torch.float32),
        },
        dict(reasons),
    )


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"V20 {label} SHA256 differs")
    return json.loads(path.read_text())


def _validate_teacher_validation(
    *,
    path: Path,
    expected_sha256: str,
    map_sha256: str,
    authorization: dict,
) -> None:
    if sha256_file(path) != expected_sha256:
        raise ValueError("V20 teacher validation SHA256 differs")
    validation = torch.load(path, map_location="cpu", weights_only=False)
    if not (
        validation.get("schema")
        == "lafgs_v19_track_extension_teacher_validation"
        and validation.get("uses_test_queries") is False
        and validation.get("loo_used") is False
        and validation.get("feedback_enters_track_registry") is False
        and validation.get("reference_available_for_novel_query") is False
        and validation.get("selection_uses_validation") is False
        and validation.get("authorization_uses_wilson_lower_bound") is True
        and validation.get("authorization_requires_independent_mapping_families")
        is True
        and validation.get("inputs", {}).get("anchor_map_sha256") == map_sha256
    ):
        raise ValueError("V20 teacher validation contract differs")
    selected = validation.get("selected_tiers", {})
    validated_authorization = {
        name: bool(item.get("authorized_actions")) for name, item in selected.items()
    }
    if validated_authorization != {
        str(name): bool(value) for name, value in authorization.items()
    }:
        raise ValueError("V20 teacher shard authorization is not validation-bound")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--observer-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--teacher-tier", choices=("tier_a", "tier_b", "tier_c"), default="tier_b")
    parser.add_argument("--minimum-wrong-winner-pose-families", type=int, default=2)
    parser.add_argument(
        "--minimum-negative-action-clean-pose-families", type=int, default=2
    )
    parser.add_argument("--maximum-repair-rows-per-query", type=int, default=256)
    parser.add_argument("--maximum-protection-rows-per-query", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    teacher_paths = [path.resolve() for path in args.teacher_shards]
    if len(set(teacher_paths)) != len(teacher_paths):
        raise ValueError("V20 teacher shard paths are duplicated")
    map_path = args.anchor_map.resolve()
    map_sha = sha256_file(map_path)
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    anchor_count = int(anchor_ids.numel())
    equivalence = torch.as_tensor(
        state.get("fine_identity_ids", torch.arange(anchor_count))
    ).long()
    if equivalence.shape != (anchor_count,):
        raise ValueError("V20 equivalence classes do not align with Anchors")

    observer_items = {}
    observer_inputs = []
    observer_shard_count = None
    observer_shard_indices = set()
    observer_certified_binding = None
    for manifest_path in args.observer_manifests:
        path = manifest_path.resolve()
        manifest = json.loads(path.read_text())
        if not (
            manifest.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
            and manifest.get("version") == 2
            and manifest.get("uses_test_queries") is False
            and manifest.get("loo_used") is False
            and manifest.get("status") == "PASS"
            and manifest.get("accepted_query_row_policy") == "v2_row_valid_only"
            and manifest.get("training_rows_are_alternative_pose_entered_only")
            is True
            and manifest.get("clean_protection_has_explicit_query_rows") is True
            and manifest.get("input", {}).get("map_sha256") == map_sha
        ):
            raise ValueError("V20 observer manifest contract differs")
        shard_count = int(manifest["shard_count"])
        shard_index = int(manifest["shard_index"])
        if observer_shard_count is None:
            observer_shard_count = shard_count
        if shard_count != observer_shard_count or not 0 <= shard_index < shard_count:
            raise ValueError("V20 observer shard counts differ")
        if shard_index in observer_shard_indices:
            raise ValueError("V20 observer shard index is duplicated")
        observer_shard_indices.add(shard_index)
        certified_binding = (
            str(Path(manifest["input"]["certified_batch"]).resolve()),
            str(manifest["input"]["certified_batch_sha256"]),
        )
        if observer_certified_binding is None:
            observer_certified_binding = certified_binding
        elif observer_certified_binding != certified_binding:
            raise ValueError("V20 observer shards use different certified batches")
        observer_inputs.append({"path": str(path), "sha256": sha256_file(path)})
        for item in manifest["records"]:
            query = int(item["query_index"])
            if query in observer_items:
                raise ValueError("V20 observer query appears in multiple manifests")
            observer_items[query] = {**item, "manifest_path": str(path)}
    if observer_shard_count != len(args.observer_manifests) or sorted(
        observer_shard_indices
    ) != list(range(int(observer_shard_count or 0))):
        raise ValueError("V20 observer shard registry is incomplete")

    design_path = args.design_batch.resolve()
    design = json.loads(design_path.read_text())
    if not (
        design.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and design.get("version") == 2
        and design.get("role") == "controller_design"
        and design.get("uses_test_queries") is False
        and design.get("loo_used") is False
        and design.get("accepted_query_row_policy") == "v2_row_valid_only"
        and design.get("training_rows_are_alternative_pose_entered_only") is True
        and design.get("clean_protection_has_explicit_query_rows") is True
        and design.get("input", {}).get("map_sha256") == map_sha
        and design.get("input", {}).get("certified_batch_sha256")
        == observer_certified_binding[1]
    ):
        raise ValueError("V20 requires the sealed controller-design split")
    expected_observer_inputs = {
        (str(Path(item["path"]).resolve()), str(item["sha256"]))
        for item in design.get("source_observer_batches", ())
    }
    actual_observer_inputs = {
        (str(Path(item["path"]).resolve()), str(item["sha256"]))
        for item in observer_inputs
    }
    if expected_observer_inputs != actual_observer_inputs:
        raise ValueError("V20 design split does not bind all observer manifests")
    design_items = {int(item["query_index"]): item for item in design["records"]}
    if len(design_items) != len(design["records"]):
        raise ValueError("V20 controller-design query IDs are not unique")
    design_family_ids = sorted(
        {int(value) for value in design.get("pose_family_ids", [])}
    )
    if (
        not design_family_ids
        or len(design_family_ids) != int(design.get("pose_family_count", -1))
    ):
        raise ValueError("V20 controller-design pose-family registry is incomplete")
    for query, item in design_items.items():
        observed = observer_items.get(query)
        if not (
            observed is not None
            and observed["sha256"] == item["sha256"]
            and Path(observed["path"]).resolve() == Path(item["path"]).resolve()
        ):
            raise ValueError("V20 design split does not bind the observer registry")

    first_teacher = torch.load(
        teacher_paths[0], map_location="cpu", weights_only=False
    )
    if not (
        first_teacher.get("schema") == "lafgs_v19_novel_track_extension_shard"
        and first_teacher.get("version") == 1
        and first_teacher.get("uses_test_queries") is False
        and first_teacher.get("loo_used") is False
        and first_teacher.get("view_role") == "feedback_query"
        and first_teacher.get("feedback_enters_track_registry") is False
        and first_teacher.get("reference_available_for_novel_query") is False
    ):
        raise ValueError("V20 requires V19 non-test feedback teacher shards")
    teacher_shard_count = int(first_teacher["shard_count"])
    if teacher_shard_count != len(teacher_paths):
        raise ValueError("V20 teacher shard registry is incomplete")
    authorization = {
        str(name): bool(value)
        for name, value in first_teacher["tier_action_authorization"].items()
    }
    if args.teacher_tier not in authorization:
        raise ValueError("V20 requested teacher tier is absent")
    teacher_inputs = first_teacher["inputs"]
    common_teacher_binding = {
        "shard_count": teacher_shard_count,
        "authorization": authorization,
        "certified_batch": str(
            Path(teacher_inputs["certified_batch"]).resolve()
        ),
        "certified_batch_sha256": str(
            teacher_inputs["certified_batch_sha256"]
        ),
        "teacher_validation": str(
            Path(teacher_inputs["teacher_validation"]).resolve()
        ),
        "teacher_validation_sha256": str(
            teacher_inputs["teacher_validation_sha256"]
        ),
        "anchor_map": str(Path(teacher_inputs["anchor_map"]).resolve()),
        "anchor_map_sha256": str(teacher_inputs["anchor_map_sha256"]),
    }
    del first_teacher
    if not (
        common_teacher_binding["anchor_map"] == str(map_path)
        and common_teacher_binding["anchor_map_sha256"] == map_sha
        and common_teacher_binding["certified_batch"]
        == observer_certified_binding[0]
        and common_teacher_binding["certified_batch_sha256"]
        == observer_certified_binding[1]
        and common_teacher_binding["certified_batch_sha256"]
        == design["input"]["certified_batch_sha256"]
    ):
        raise ValueError("V20 teacher/observer/design lineage differs")
    certified_path = Path(common_teacher_binding["certified_batch"])
    certified = _load_bound_json(
        certified_path,
        common_teacher_binding["certified_batch_sha256"],
        "certified batch",
    )
    if not (
        certified.get("schema")
        in {
            "lafgs_v7_certified_clean_render_batch",
            "lafgs_v13_merged_certified_render_batch",
            "lafgs_v14_observer_split_certified_view",
        }
        and certified.get("view_role") == "feedback_query"
        and certified.get("uses_test_queries") is False
        and certified.get("map_mutation_count") == 0
    ):
        raise ValueError("V20 certified feedback batch contract differs")
    certified_items = {}
    for item in certified["records"]:
        query = int(item["query_index"])
        if query in certified_items:
            raise ValueError("V20 certified feedback query is duplicated")
        certified_items[query] = item
    expected_teacher_queries = {
        query
        for query, item in certified_items.items()
        if item.get("decision") == "ACCEPT"
    }
    if set(observer_items) != set(certified_items):
        raise ValueError("V20 observer registry does not cover the certified batch")
    design_source_record_sha256s = sorted(
        str(certified_items[query]["sha256"]) for query in design_items
    )
    if (
        len(design_source_record_sha256s) != len(design_items)
        or len(set(design_source_record_sha256s))
        != len(design_source_record_sha256s)
        or any(len(value) != 64 for value in design_source_record_sha256s)
    ):
        raise ValueError("V20 design source-record registry is invalid")
    _validate_teacher_validation(
        path=Path(common_teacher_binding["teacher_validation"]),
        expected_sha256=common_teacher_binding["teacher_validation_sha256"],
        map_sha256=map_sha,
        authorization=authorization,
    )

    records = []
    excluded_invalid_rows = 0
    protection_queries = []
    protection_positive: list[torch.Tensor] = []
    protection_negative: list[torch.Tensor] = []
    protection_margin = []
    protection_query_indices = []
    protection_source_rows = []
    training_binding_counts: Counter[str] = Counter()
    protection_binding_counts: Counter[str] = Counter()
    teacher_shard_indices = set()
    teacher_queries = set()
    teacher_artifact_inputs = []
    for teacher_path in teacher_paths:
        teacher_sha = sha256_file(teacher_path)
        teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
        inputs = teacher.get("inputs", {})
        binding = {
            "shard_count": int(teacher.get("shard_count", -1)),
            "authorization": {
                str(name): bool(value)
                for name, value in teacher.get(
                    "tier_action_authorization", {}
                ).items()
            },
            "certified_batch": str(
                Path(inputs.get("certified_batch", ".")).resolve()
            ),
            "certified_batch_sha256": str(
                inputs.get("certified_batch_sha256", "")
            ),
            "teacher_validation": str(
                Path(inputs.get("teacher_validation", ".")).resolve()
            ),
            "teacher_validation_sha256": str(
                inputs.get("teacher_validation_sha256", "")
            ),
            "anchor_map": str(Path(inputs.get("anchor_map", ".")).resolve()),
            "anchor_map_sha256": str(inputs.get("anchor_map_sha256", "")),
        }
        if not (
            teacher.get("schema") == "lafgs_v19_novel_track_extension_shard"
            and teacher.get("version") == 1
            and teacher.get("uses_test_queries") is False
            and teacher.get("loo_used") is False
            and teacher.get("view_role") == "feedback_query"
            and teacher.get("feedback_enters_track_registry") is False
            and teacher.get("reference_available_for_novel_query") is False
            and binding == common_teacher_binding
        ):
            raise ValueError("V20 teacher shard contracts or lineage differ")
        shard_index = int(teacher["shard_index"])
        if not 0 <= shard_index < teacher_shard_count:
            raise ValueError("V20 teacher shard index is invalid")
        if shard_index in teacher_shard_indices:
            raise ValueError("V20 teacher shard index is duplicated")
        teacher_shard_indices.add(shard_index)
        teacher_artifact_inputs.append(
            {"path": str(teacher_path), "sha256": teacher_sha}
        )
        for item in teacher["records"]:
            query = int(item["query_index"])
            if query in teacher_queries:
                raise ValueError("V20 teacher query appears in multiple shards")
            teacher_queries.add(query)
            certified_item = certified_items.get(query)
            if not (
                certified_item is not None
                and certified_item.get("decision") == "ACCEPT"
                and Path(certified_item["path"]).resolve()
                == Path(item["source_record"]).resolve()
                and certified_item["sha256"] == item["source_record_sha256"]
            ):
                raise ValueError("V20 teacher query is not certified-batch-bound")
            observer_item = observer_items.get(query)
            if observer_item is None:
                raise ValueError("V20 teacher query is missing its observer record")
            observer_path = Path(observer_item["path"]).resolve()
            if sha256_file(observer_path) != observer_item["sha256"]:
                raise ValueError("V20 observer record SHA256 differs")
            observer = torch.load(
                observer_path, map_location="cpu", weights_only=False
            )
            if not (
                observer.get("schema") == "lafgs_v9_no_loo_causal_feedback_record"
                and observer.get("version") == 2
                and observer.get("loo_used") is False
                and observer.get("query_index") == query
                and bool(observer.get("can_train_metric"))
                == bool(observer_item.get("can_train_metric"))
                and int(observer["pose_family_id"]) == int(item["pose_family_id"])
                and str(Path(observer["source_record"]).resolve())
                == str(Path(item["source_record"]).resolve())
                and observer["source_record_sha256"]
                == item["source_record_sha256"]
            ):
                raise ValueError("V20 teacher and observer source bindings differ")
            teacher_rows = torch.as_tensor(item["source_query_rows"]).long()
            observer_rows = torch.as_tensor(observer["source_query_rows"]).long()
            if not torch.equal(teacher_rows, observer_rows):
                raise ValueError("V20 teacher and observer valid-row registries differ")
            if query not in design_items:
                continue
            truth = item["truth_tiers"][args.teacher_tier]
            training = observer["training_evidence"]
            local_rows = torch.as_tensor(training["query_rows"]).long()
            training_descriptors = torch.as_tensor(
                training["query_descriptors"]
            ).float()
            excluded_invalid_rows += int(observer.get("invalid_source_row_count", 0))
            clean = observer["clean_protection_evidence"]
            topk_rows = torch.as_tensor(observer["topk_anchor_rows"]).long()
            topk_scores = torch.as_tensor(observer["topk_scores"]).float()
            if topk_rows.shape[0] != teacher_rows.numel():
                raise ValueError("V20 observer Top-K does not align with valid rows")
            clean_bound, clean_counts = _truth_bound_clean_protection(
                clean=clean,
                candidate_anchor_rows=topk_rows,
                candidate_scores=topk_scores,
                truth=truth,
                equivalence_class_ids=equivalence,
            )
            protection_binding_counts.update(clean_counts)
            if clean_bound["query_descriptors"].shape[0]:
                count = clean_bound["query_descriptors"].shape[0]
                clean_rows = clean_bound["query_rows"]
                clean_truth = _subset_truth(truth, clean_rows)
                clean_policy = partition_feedback_rows(
                    row_valid=torch.ones(count, dtype=torch.bool),
                    truth_status=clean_truth["truth_status"],
                )
                # Feed truth-bound clean rows through the competition builder as
                # well as the hard protection CSR.  This is what proves whether
                # a recurrent wrong winner has an observable clean role across
                # pose families before a negative-side Anchor action is allowed.
                records.append(
                    {
                        "query_index": query,
                        "pose_family_id": int(item["pose_family_id"]),
                        "can_train_descriptor": False,
                        "actual_query_task_gain": 0.0,
                        "query_descriptors": clean_bound[
                            "query_descriptors"
                        ],
                        "source_query_rows": clean_rows,
                        "candidate_anchor_rows": topk_rows[clean_rows],
                        "candidate_scores": topk_scores[clean_rows],
                        "truth": clean_truth,
                        "row_policy": clean_policy,
                    }
                )
                protection_queries.append(clean_bound["query_descriptors"])
                protection_positive.extend(clean_bound["positive_anchor_rows"])
                protection_negative.extend(clean_bound["negative_anchor_rows"])
                protection_margin.append(clean_bound["initial_margin"])
                protection_query_indices.append(
                    torch.full((count,), query, dtype=torch.long)
                )
                protection_source_rows.append(clean_bound["query_rows"])
            if not bool(observer["can_train_metric"]):
                continue
            if local_rows.numel() == 0 or training_descriptors.shape[0] == 0:
                continue
            task_gain = float(observer["actual_task_gain"])
            training_gain = float(training["actual_query_task_gain"])
            if not (
                torch.isfinite(torch.tensor(task_gain))
                and abs(task_gain - training_gain) <= 1e-6
            ):
                raise ValueError("V20 observer task-gain binding differs")
            bound, binding_counts = _bind_v9_training_rows(
                training=training,
                candidate_anchor_rows=topk_rows,
                candidate_scores=topk_scores,
                truth=truth,
                equivalence_class_ids=equivalence,
            )
            training_binding_counts.update(binding_counts)
            if not bool(bound.any()):
                continue
            local_rows = local_rows[bound]
            training_descriptors = training_descriptors[bound]
            local_truth = _subset_truth(truth, local_rows)
            row_count = int(local_rows.numel())
            policy = partition_feedback_rows(
                row_valid=torch.ones(row_count, dtype=torch.bool),
                truth_status=local_truth["truth_status"],
            )
            records.append(
                {
                    "query_index": query,
                    "pose_family_id": int(item["pose_family_id"]),
                    "can_train_descriptor": bool(observer["can_train_metric"]),
                    "actual_query_task_gain": task_gain,
                    "query_descriptors": training_descriptors,
                    "source_query_rows": local_rows,
                    "candidate_anchor_rows": topk_rows[local_rows],
                    "candidate_scores": topk_scores[local_rows],
                    "truth": local_truth,
                    "row_policy": policy,
                }
            )
        del teacher
    if sorted(teacher_shard_indices) != list(range(teacher_shard_count)):
        raise ValueError("V20 teacher shard registry is incomplete")
    if teacher_queries != expected_teacher_queries:
        raise ValueError("V20 teacher shards do not exactly cover certified ACCEPT queries")

    if not protection_queries:
        raise RuntimeError(
            "no clean protection row has explicit Query-row and V19 truth binding"
        )
    evidence = build_topk_competition_evidence(
        records=records,
        anchor_count=anchor_count,
        equivalence_class_ids=equivalence,
        minimum_wrong_winner_pose_families=int(
            args.minimum_wrong_winner_pose_families
        ),
        minimum_negative_action_clean_pose_families=int(
            args.minimum_negative_action_clean_pose_families
        ),
        maximum_repair_rows_per_query=int(args.maximum_repair_rows_per_query),
        maximum_protection_rows_per_query=int(args.maximum_protection_rows_per_query),
    )
    query = torch.cat(protection_queries)
    positive_offsets, positive = _csr(protection_positive)
    negative_offsets, negative = _csr(protection_negative)
    margin = torch.cat(protection_margin)
    if not (
        query.shape[0]
        == positive_offsets.numel() - 1
        == negative_offsets.numel() - 1
        == margin.numel()
    ):
        raise RuntimeError("V20 clean protection materialization is misaligned")
    evidence.update(
        {
            "protection_query_descriptors": query,
            "protection_positive_offsets": positive_offsets,
            "protection_positive_anchor_rows": positive,
            "protection_negative_offsets": negative_offsets,
            "protection_negative_anchor_rows": negative,
            "protection_initial_margin": margin,
            "protection_query_indices": torch.cat(protection_query_indices),
            "protection_source_query_rows": torch.cat(protection_source_rows),
            "protection_source": (
                "v9_explicit_query_row_v19_truth_bound_current_winner_topk_nonpositives"
            ),
        }
    )
    evidence.update(
        {
            "teacher_tier": args.teacher_tier,
            "teacher_tier_action_authorized": bool(
                authorization[args.teacher_tier]
            ),
            "strong_feedback_authorized": bool(
                args.teacher_tier == "tier_b"
                and authorization[args.teacher_tier]
            ),
            "accepted_query_row_policy": "v2_row_valid_only",
            "repair_row_binding_contract": (
                "v9_rank_pair_and_current_winner_confirmed_by_v19_truth"
            ),
            "clean_protection_binding_contract": (
                "explicit_query_row_current_winner_and_topk_nonpositives_confirmed_by_v19_truth"
            ),
            "design_query_indices": sorted(design_items),
            "design_pose_family_ids": design_family_ids,
            "design_source_record_sha256s": design_source_record_sha256s,
            "v9_v19_training_binding_counts": dict(training_binding_counts),
            "clean_protection_binding_counts": dict(
                protection_binding_counts
            ),
            "invalid_rows_retained_by_plant_but_excluded_from_repair": int(
                excluded_invalid_rows
            ),
            "inputs": {
                "anchor_map": str(map_path),
                "anchor_map_sha256": map_sha,
                "teacher_shards": teacher_artifact_inputs,
                "teacher_validation": common_teacher_binding[
                    "teacher_validation"
                ],
                "teacher_validation_sha256": common_teacher_binding[
                    "teacher_validation_sha256"
                ],
                "certified_batch": common_teacher_binding[
                    "certified_batch"
                ],
                "certified_batch_sha256": common_teacher_binding[
                    "certified_batch_sha256"
                ],
                "observer_manifests": observer_inputs,
                "design_batch": str(design_path),
                "design_batch_sha256": sha256_file(design_path),
            },
        }
    )
    _save(evidence, args.output.resolve())
    summary = {
        "schema": evidence["schema"],
        "teacher_tier": args.teacher_tier,
        "strong_feedback_authorized": evidence["strong_feedback_authorized"],
        "repair_row_count": int(evidence["repair_query_descriptors"].shape[0]),
        "protection_row_count": int(
            evidence["protection_query_descriptors"].shape[0]
        ),
        "counts": evidence["counts"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
