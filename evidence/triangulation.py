from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from evidence.camera_pair_policy import (
    _camera_centers_and_axes,
    _camera_pair_geometry_table,
    candidate_camera_pairs,
)


def camera_pose_bins(
    pose_w2c: torch.Tensor,
    bin_count: int,
    *,
    direction_weight: float = 0.5,
) -> torch.Tensor:
    """Assign cameras to deterministic center-and-view-direction bins."""
    camera_centers, optical_axis = _camera_centers_and_axes(pose_w2c)
    count = int(camera_centers.shape[0])
    bin_count = min(max(int(bin_count), 1), count)
    if count == 0:
        return torch.zeros(0, dtype=torch.long)
    centered = camera_centers - camera_centers.median(dim=0).values
    positive_radius = torch.linalg.norm(centered, dim=1)
    scale = positive_radius[positive_radius > 0].median().clamp_min(1e-6)
    embedding = torch.cat(
        (centered / scale, optical_axis * float(direction_weight)), dim=1
    )
    centroid = embedding.mean(dim=0)
    first = int(
        torch.argmax((embedding - centroid).square().sum(dim=1)).item()
    )
    selected = [first]
    selected_mask = torch.zeros(count, dtype=torch.bool)
    selected_mask[first] = True
    nearest = (embedding - embedding[first]).square().sum(dim=1)
    for _ in range(1, bin_count):
        index = int(
            torch.argmax(nearest.masked_fill(selected_mask, -torch.inf)).item()
        )
        selected.append(index)
        selected_mask[index] = True
        nearest = torch.minimum(
            nearest,
            (embedding - embedding[index]).square().sum(dim=1),
        )
    prototypes = embedding[torch.as_tensor(selected, dtype=torch.long)]
    return torch.cdist(embedding, prototypes).argmin(dim=1)


def camera_center_bins(pose_w2c: torch.Tensor, bin_count: int) -> torch.Tensor:
    """Legacy center-only camera bins retained for reproducibility."""
    return camera_pose_bins(pose_w2c, bin_count, direction_weight=0.0)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )


def fundamental_from_known_poses(
    K_a: torch.Tensor,
    pose_a_w2c: torch.Tensor,
    K_b: torch.Tensor,
    pose_b_w2c: torch.Tensor,
) -> torch.Tensor:
    """Return F such that x_b.T F x_a = 0 for physical pixel coordinates."""
    K_a = torch.as_tensor(K_a, dtype=torch.float64)
    K_b = torch.as_tensor(K_b, dtype=torch.float64)
    pose_a_w2c = torch.as_tensor(pose_a_w2c, dtype=torch.float64)
    pose_b_w2c = torch.as_tensor(pose_b_w2c, dtype=torch.float64)
    rotation = pose_b_w2c[:3, :3] @ pose_a_w2c[:3, :3].T
    translation = (
        pose_b_w2c[:3, 3] - rotation @ pose_a_w2c[:3, 3]
    )
    essential = _skew(translation) @ rotation
    return torch.linalg.inv(K_b).T @ essential @ torch.linalg.inv(K_a)


def symmetric_epipolar_distance(
    uv_a: torch.Tensor,
    uv_b: torch.Tensor,
    fundamental: torch.Tensor,
) -> torch.Tensor:
    """Compute symmetric point-to-epipolar-line distance in pixels."""
    uv_a = torch.as_tensor(uv_a, dtype=torch.float64)
    uv_b = torch.as_tensor(
        uv_b, device=uv_a.device, dtype=torch.float64
    )
    fundamental = torch.as_tensor(
        fundamental, device=uv_a.device, dtype=torch.float64
    )
    ones = torch.ones(
        (uv_a.shape[0], 1), device=uv_a.device, dtype=uv_a.dtype
    )
    point_a = torch.cat((uv_a, ones), dim=1)
    point_b = torch.cat((uv_b, ones), dim=1)
    line_b = point_a @ fundamental.T
    line_a = point_b @ fundamental
    numerator = (point_b * line_b).sum(dim=1).abs()
    distance_a = numerator / torch.linalg.norm(line_a[:, :2], dim=1).clamp_min(
        1e-12
    )
    distance_b = numerator / torch.linalg.norm(line_b[:, :2], dim=1).clamp_min(
        1e-12
    )
    return 0.5 * (distance_a + distance_b)


@torch.no_grad()
def reciprocal_epipolar_matches(
    descriptors_a: torch.Tensor,
    descriptors_b: torch.Tensor,
    uv_a: torch.Tensor,
    uv_b: torch.Tensor,
    K_a: torch.Tensor,
    pose_a_w2c: torch.Tensor,
    K_b: torch.Tensor,
    pose_b_w2c: torch.Tensor,
    *,
    minimum_similarity: float = 0.65,
    minimum_margin: float = 0.01,
    maximum_epipolar_error_px: float = 2.0,
    epipolar_candidate_topk: int = 1,
    recovered_minimum_similarity: float = -1.0,
    recovered_minimum_margin: float = -1.0,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]
]:
    """Match descriptors with optional epipolar-first top-K reciprocity."""
    def returned(source, target, confidence, diagnostics=None):
        base = (source, target, confidence)
        if not bool(return_diagnostics):
            return base
        return (*base, dict(diagnostics or {}))

    if descriptors_a.numel() == 0 or descriptors_b.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return returned(
            empty,
            empty,
            torch.empty(0),
            {
                "source_keypoint_count": int(descriptors_a.shape[0]),
                "target_keypoint_count": int(descriptors_b.shape[0]),
                "raw_top1_reciprocal_count": 0,
                "descriptor_accepted_before_epipolar_count": 0,
                "epipolar_accepted_top1_count": 0,
                "epipolar_rejected_after_descriptor_count": 0,
                "ambiguity_rejected_count": 0,
                "final_reciprocal_epipolar_count": 0,
                "epipolar_recovered_final_count": 0,
            },
        )
    descriptors_a = F.normalize(descriptors_a.float(), dim=1)
    descriptors_b = F.normalize(descriptors_b.float(), dim=1)
    similarity = descriptors_a @ descriptors_b.T
    candidate_topk = int(epipolar_candidate_topk)
    if candidate_topk > 1:
        candidate_topk = min(
            candidate_topk,
            int(similarity.shape[0]),
            int(similarity.shape[1]),
        )
        if candidate_topk < 2:
            candidate_topk = 1
    if candidate_topk > 1:
        fundamental = fundamental_from_known_poses(
            K_a, pose_a_w2c, K_b, pose_b_w2c
        ).to(device=similarity.device)
        uv_a_device = torch.as_tensor(
            uv_a, device=similarity.device, dtype=torch.float64
        )
        uv_b_device = torch.as_tensor(
            uv_b, device=similarity.device, dtype=torch.float64
        )

        values_ab, indices_ab = torch.topk(
            similarity, k=candidate_topk, dim=1
        )
        epipolar_ab = symmetric_epipolar_distance(
            uv_a_device[:, None, :]
            .expand(-1, candidate_topk, -1)
            .reshape(-1, 2),
            uv_b_device[indices_ab].reshape(-1, 2),
            fundamental,
        ).reshape_as(values_ab)
        valid_ab = epipolar_ab <= float(maximum_epipolar_error_px)
        gated_ab = values_ab.masked_fill(~valid_ab, -torch.inf)
        best_ab, best_ab_position = torch.topk(gated_ab, k=2, dim=1)
        target = indices_ab.gather(
            1, best_ab_position[:, :1]
        ).squeeze(1)
        chosen_epipolar = epipolar_ab.gather(
            1, best_ab_position[:, :1]
        ).squeeze(1)

        values_ba, indices_ba = torch.topk(
            similarity, k=candidate_topk, dim=0
        )
        values_ba = values_ba.T
        indices_ba = indices_ba.T
        epipolar_ba = symmetric_epipolar_distance(
            uv_a_device[indices_ba].reshape(-1, 2),
            uv_b_device[:, None, :]
            .expand(-1, candidate_topk, -1)
            .reshape(-1, 2),
            fundamental,
        ).reshape_as(values_ba)
        valid_ba = epipolar_ba <= float(maximum_epipolar_error_px)
        gated_ba = values_ba.masked_fill(~valid_ba, -torch.inf)
        best_ba, best_ba_position = torch.topk(gated_ba, k=2, dim=1)
        source_for_target = indices_ba.gather(
            1, best_ba_position[:, :1]
        ).squeeze(1)

        source = torch.arange(
            descriptors_a.shape[0],
            device=similarity.device,
            dtype=torch.long,
        )
        reciprocal = source_for_target[target] == source
        recovered = (
            (target != indices_ab[:, 0])
            | (source != indices_ba[target, 0])
        )
        recovered_similarity = (
            float(recovered_minimum_similarity)
            if float(recovered_minimum_similarity) >= 0.0
            else float(minimum_similarity)
        )
        recovered_margin = (
            float(recovered_minimum_margin)
            if float(recovered_minimum_margin) >= 0.0
            else float(minimum_margin)
        )
        recovered_valid = (
            (best_ab[:, 0] >= recovered_similarity)
            & ((best_ab[:, 0] - best_ab[:, 1]) >= recovered_margin)
            & (
                (best_ba[target, 0] - best_ba[target, 1])
                >= recovered_margin
            )
        )
        descriptor_valid = (
            reciprocal
            & torch.isfinite(best_ab[:, 0])
            & torch.isfinite(best_ba[target, 0])
            & (best_ab[:, 0] >= float(minimum_similarity))
            & ((best_ab[:, 0] - best_ab[:, 1]) >= float(minimum_margin))
            & (
                (best_ba[target, 0] - best_ba[target, 1])
                >= float(minimum_margin)
            )
            & (~recovered | recovered_valid)
        )
        selected = torch.nonzero(
            descriptor_valid, as_tuple=False
        ).reshape(-1)
        diagnostics = None
        if bool(return_diagnostics):
            raw_target = indices_ab[:, 0]
            raw_reciprocal = indices_ba[raw_target, 0] == source
            raw_margin_ab = values_ab[:, 0] - values_ab[:, 1]
            raw_margin_ba = (
                values_ba[raw_target, 0] - values_ba[raw_target, 1]
            )
            raw_descriptor_valid = (
                raw_reciprocal
                & (values_ab[:, 0] >= float(minimum_similarity))
                & (raw_margin_ab >= float(minimum_margin))
                & (raw_margin_ba >= float(minimum_margin))
            )
            raw_selected = torch.nonzero(
                raw_descriptor_valid, as_tuple=False
            ).reshape(-1)
            if raw_selected.numel():
                raw_epipolar = symmetric_epipolar_distance(
                    torch.as_tensor(
                        uv_a, device=similarity.device, dtype=torch.float64
                    )[raw_selected],
                    torch.as_tensor(
                        uv_b, device=similarity.device, dtype=torch.float64
                    )[raw_target[raw_selected]],
                    fundamental,
                )
                raw_epipolar_accepted_tensor = (
                    raw_epipolar <= float(maximum_epipolar_error_px)
                ).sum()
            else:
                raw_epipolar_accepted_tensor = source.new_zeros(())
            finite_reciprocal = reciprocal & torch.isfinite(best_ab[:, 0]) & torch.isfinite(
                best_ba[target, 0]
            )
            counts = torch.stack(
                (
                    raw_reciprocal.sum(),
                    raw_descriptor_valid.sum(),
                    raw_epipolar_accepted_tensor.to(source.device),
                    finite_reciprocal.sum(),
                    descriptor_valid.sum(),
                    recovered[selected].sum() if selected.numel() else source.new_zeros(()),
                    (~valid_ab).sum(),
                    (~valid_ba).sum(),
                    (~valid_ab.any(dim=1)).sum(),
                )
            ).detach().cpu().tolist()
            (
                raw_reciprocal_count,
                raw_descriptor_count,
                raw_epipolar_accepted,
                finite_reciprocal_count,
                descriptor_valid_count,
                recovered_selected_count,
                rejected_ab_count,
                rejected_ba_count,
                missing_source_count,
            ) = (int(value) for value in counts)
            diagnostics = {
                "source_keypoint_count": int(descriptors_a.shape[0]),
                "target_keypoint_count": int(descriptors_b.shape[0]),
                "raw_top1_reciprocal_count": raw_reciprocal_count,
                "descriptor_accepted_before_epipolar_count": raw_descriptor_count,
                "epipolar_accepted_top1_count": raw_epipolar_accepted,
                "epipolar_rejected_after_descriptor_count": raw_descriptor_count
                - raw_epipolar_accepted,
                "ambiguity_rejected_count": finite_reciprocal_count
                - descriptor_valid_count,
                "final_reciprocal_epipolar_count": int(selected.numel()),
                "epipolar_recovered_final_count": recovered_selected_count,
                "epipolar_topk_rejected_directed_candidate_count": (
                    rejected_ab_count + rejected_ba_count
                ),
                "source_without_epipolar_candidate_count": missing_source_count,
            }
        if selected.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            return returned(empty, empty, torch.empty(0), diagnostics)
        selected_target = target[selected]
        selected_epipolar = chosen_epipolar[selected].float()
        confidence = best_ab[selected, 0].detach().float() * torch.exp(
            -0.5
            * (
                selected_epipolar
                / max(float(maximum_epipolar_error_px), 1e-6)
            ).square()
        )
        return returned(
            selected.detach().cpu().long(),
            selected_target.detach().cpu().long(),
            confidence.detach().cpu(),
            diagnostics,
        )

    values_ab, indices_ab = torch.topk(similarity, k=2, dim=1)
    values_ba, indices_ba = torch.topk(similarity, k=2, dim=0)
    source = torch.arange(
        descriptors_a.shape[0], device=similarity.device, dtype=torch.long
    )
    target = indices_ab[:, 0]
    reciprocal = indices_ba[0, target] == source
    margin_ab = values_ab[:, 0] - values_ab[:, 1]
    margin_ba = values_ba[0, target] - values_ba[1, target]
    descriptor_valid = (
        reciprocal
        & (values_ab[:, 0] >= float(minimum_similarity))
        & (margin_ab >= float(minimum_margin))
        & (margin_ba >= float(minimum_margin))
    )
    selected = torch.nonzero(descriptor_valid, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        diagnostics = {
            "source_keypoint_count": int(descriptors_a.shape[0]),
            "target_keypoint_count": int(descriptors_b.shape[0]),
            "raw_top1_reciprocal_count": int(reciprocal.sum()),
            "descriptor_accepted_before_epipolar_count": 0,
            "epipolar_accepted_top1_count": 0,
            "epipolar_rejected_after_descriptor_count": 0,
            "ambiguity_rejected_count": int(reciprocal.sum()),
            "final_reciprocal_epipolar_count": 0,
            "epipolar_recovered_final_count": 0,
        }
        return returned(empty, empty, torch.empty(0), diagnostics)
    selected_target = target[selected]
    selected_cpu = selected.detach().cpu()
    selected_target_cpu = selected_target.detach().cpu()
    fundamental = fundamental_from_known_poses(
        K_a, pose_a_w2c, K_b, pose_b_w2c
    )
    epipolar = symmetric_epipolar_distance(
        torch.as_tensor(uv_a).cpu()[selected_cpu],
        torch.as_tensor(uv_b).cpu()[selected_target_cpu],
        fundamental,
    )
    epipolar_valid = epipolar <= float(maximum_epipolar_error_px)
    selected = selected_cpu[epipolar_valid]
    selected_target = selected_target_cpu[epipolar_valid]
    confidence = (
        values_ab[selected.to(values_ab.device), 0].detach().float().cpu()
        * torch.exp(
            -0.5
            * (
                epipolar[epipolar_valid].float()
                / max(float(maximum_epipolar_error_px), 1e-6)
            ).square()
        )
    )
    accepted = int(epipolar_valid.sum())
    diagnostics = {
        "source_keypoint_count": int(descriptors_a.shape[0]),
        "target_keypoint_count": int(descriptors_b.shape[0]),
        "raw_top1_reciprocal_count": int(reciprocal.sum()),
        "descriptor_accepted_before_epipolar_count": int(
            descriptor_valid.sum()
        ),
        "epipolar_accepted_top1_count": accepted,
        "epipolar_rejected_after_descriptor_count": int(
            descriptor_valid.sum()
        )
        - accepted,
        "ambiguity_rejected_count": int(reciprocal.sum())
        - int(descriptor_valid.sum()),
        "final_reciprocal_epipolar_count": accepted,
        "epipolar_recovered_final_count": 0,
    }
    return returned(selected.long(), selected_target.long(), confidence, diagnostics)


def _cycle_supported_pair_edges(
    pair_matches: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    keypoint_counts: list[int],
) -> dict[tuple[int, int], torch.Tensor]:
    """Mark pair edges that participate in a descriptor-consistent 3-cycle."""
    supported = {
        pair: torch.zeros(match[0].numel(), dtype=torch.bool)
        for pair, match in pair_matches.items()
    }
    neighbors = defaultdict(set)
    for left, right in pair_matches:
        neighbors[left].add(right)
        neighbors[right].add(left)
    cameras = sorted(neighbors)
    for left in cameras:
        right_candidates = sorted(index for index in neighbors[left] if index > left)
        for position, middle in enumerate(right_candidates):
            for right in right_candidates[position + 1 :]:
                pair_lm = (left, middle)
                pair_lr = (left, right)
                pair_mr = (middle, right)
                if pair_mr not in pair_matches:
                    continue
                lm_left, lm_middle, _ = pair_matches[pair_lm]
                lr_left, lr_right, _ = pair_matches[pair_lr]
                mr_middle, mr_right, _ = pair_matches[pair_mr]
                lookup_lr = torch.full(
                    (keypoint_counts[left],), -1, dtype=torch.long
                )
                edge_lr = torch.full_like(lookup_lr, -1)
                lookup_lr[lr_left] = lr_right
                edge_lr[lr_left] = torch.arange(lr_left.numel())
                lookup_mr = torch.full(
                    (keypoint_counts[middle],), -1, dtype=torch.long
                )
                edge_mr = torch.full_like(lookup_mr, -1)
                lookup_mr[mr_middle] = mr_right
                edge_mr[mr_middle] = torch.arange(mr_middle.numel())
                cycle_right = lookup_lr[lm_left]
                cycle_mr_edge = edge_mr[lm_middle]
                cycle = (
                    (cycle_right >= 0)
                    & (cycle_mr_edge >= 0)
                    & (lookup_mr[lm_middle] == cycle_right)
                )
                if not bool(cycle.any()):
                    continue
                lm_edges = torch.nonzero(cycle, as_tuple=False).reshape(-1)
                supported[pair_lm][lm_edges] = True
                supported[pair_lr][edge_lr[lm_left[lm_edges]]] = True
                supported[pair_mr][cycle_mr_edge[lm_edges]] = True
    return supported


class _SparseDisjointSet:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_rank = self.rank.get(left_root, 0)
        right_rank = self.rank.get(right_root, 0)
        if left_rank < right_rank:
            left_root, right_root = right_root, left_root
            left_rank, right_rank = right_rank, left_rank
        self.parent[right_root] = left_root
        if left_rank == right_rank:
            self.rank[left_root] = left_rank + 1


class _ConflictAwareTrackSet:
    """Disjoint set that preserves at most one keypoint per query."""

    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.queries = {}
        self.cycle_seeded = {}

    def add(self, node, query):
        if node in self.parent:
            return
        self.parent[node] = node
        self.rank[node] = 0
        self.queries[node] = {int(query)}
        self.cycle_seeded[node] = False

    def find(self, node):
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left, right, *, cycle_supported):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            if cycle_supported:
                self.cycle_seeded[left_root] = True
            return True
        if self.queries[left_root] & self.queries[right_root]:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        self.queries[left_root].update(self.queries.pop(right_root))
        self.cycle_seeded[left_root] = (
            self.cycle_seeded[left_root]
            or self.cycle_seeded.pop(right_root)
            or bool(cycle_supported)
        )
        return True


def _graded_track_components(
    pair_matches,
    cycle_support,
    keypoint_offsets,
):
    """Build cycle-seeded and pure-chain tracks without query collisions."""
    edge_left = []
    edge_right = []
    edge_confidence = []
    edge_cycle = []
    edge_source = []
    edge_target = []
    edge_pair_position = []
    edge_accepted = {
        pair: torch.zeros(match[0].numel(), dtype=torch.bool)
        for pair, match in pair_matches.items()
    }
    edge_conflict_rejected = {
        pair: torch.zeros(match[0].numel(), dtype=torch.bool)
        for pair, match in pair_matches.items()
    }
    for pair, (source, target, confidence) in pair_matches.items():
        left, right = pair
        count = int(source.numel())
        if count == 0:
            continue
        edge_left.extend([left] * count)
        edge_right.extend([right] * count)
        edge_source.append(source.long())
        edge_target.append(target.long())
        edge_confidence.append(confidence.float())
        edge_cycle.append(cycle_support[pair].bool())
        edge_pair_position.append(torch.arange(count, dtype=torch.long))
    if not edge_source:
        return (
            {},
            {},
            {},
            {
                "track_graded_cycle_edge_count": 0,
                "track_graded_chain_edge_count": 0,
                "track_graded_conflict_rejected_edge_count": 0,
            },
            {"accepted": edge_accepted, "conflict_rejected": edge_conflict_rejected},
        )
    edge_source = torch.cat(edge_source)
    edge_target = torch.cat(edge_target)
    edge_confidence = torch.cat(edge_confidence)
    edge_cycle = torch.cat(edge_cycle)
    edge_pair_position = torch.cat(edge_pair_position)
    edge_left = torch.as_tensor(edge_left, dtype=torch.long)
    edge_right = torch.as_tensor(edge_right, dtype=torch.long)
    cycle_indices = torch.nonzero(edge_cycle, as_tuple=False).reshape(-1)
    chain_indices = torch.nonzero(~edge_cycle, as_tuple=False).reshape(-1)
    if cycle_indices.numel():
        cycle_indices = cycle_indices[
            torch.argsort(
                edge_confidence[cycle_indices],
                descending=True,
                stable=True,
            )
        ]
    if chain_indices.numel():
        chain_indices = chain_indices[
            torch.argsort(
                edge_confidence[chain_indices],
                descending=True,
                stable=True,
            )
        ]
    order = torch.cat((cycle_indices, chain_indices))
    disjoint = _ConflictAwareTrackSet()
    node_confidence = defaultdict(float)
    accepted_cycle = 0
    accepted_chain = 0
    rejected_conflict = 0
    for edge in order.tolist():
        left_query = int(edge_left[edge])
        right_query = int(edge_right[edge])
        source_node = (
            int(keypoint_offsets[left_query]) + int(edge_source[edge])
        )
        target_node = (
            int(keypoint_offsets[right_query]) + int(edge_target[edge])
        )
        disjoint.add(source_node, left_query)
        disjoint.add(target_node, right_query)
        is_cycle = bool(edge_cycle[edge])
        pair = (left_query, right_query)
        pair_position = int(edge_pair_position[edge])
        if not disjoint.union(
            source_node, target_node, cycle_supported=is_cycle
        ):
            rejected_conflict += 1
            edge_conflict_rejected[pair][pair_position] = True
            continue
        edge_accepted[pair][pair_position] = True
        confidence = float(edge_confidence[edge])
        node_confidence[source_node] = max(
            node_confidence[source_node], confidence
        )
        node_confidence[target_node] = max(
            node_confidence[target_node], confidence
        )
        if is_cycle:
            accepted_cycle += 1
        else:
            accepted_chain += 1
    components = defaultdict(list)
    for node in disjoint.parent:
        components[disjoint.find(node)].append(node)
    component_cycle_seeded = {
        root: bool(disjoint.cycle_seeded[disjoint.find(root)])
        for root in components
    }
    diagnostics = {
        "track_graded_cycle_edge_count": accepted_cycle,
        "track_graded_chain_edge_count": accepted_chain,
        "track_graded_conflict_rejected_edge_count": rejected_conflict,
    }
    return (
        components,
        node_confidence,
        component_cycle_seeded,
        diagnostics,
        {"accepted": edge_accepted, "conflict_rejected": edge_conflict_rejected},
    )


@torch.no_grad()
def build_cycle_consistent_tracks(
    *,
    descriptors: list[torch.Tensor],
    keypoints: list[torch.Tensor],
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    detector_scores: list[torch.Tensor] | None = None,
    pair_neighbors: int = 6,
    pair_policy: str = "nearest",
    pair_budget: int | None = None,
    pair_image_hw: torch.Tensor | None = None,
    pair_scene_points_xyz: torch.Tensor | None = None,
    pair_minimum_overlap_jaccard: float = 0.15,
    pair_minimum_joint_visibility_points: int = 8,
    pair_parallax_saturation_deg: float = 2.0,
    pair_diversity_weight: float = 0.20,
    pair_candidate_pool_per_camera: int = 48,
    pair_scene_depth_m: torch.Tensor | None = None,
    pair_minimum_expected_parallax_deg: float = 1.0,
    pair_near_fraction: float = 1.0 / 3.0,
    pair_maximum_baseline_depth_ratio: float = 0.5,
    minimum_baseline_m: float = 0.03,
    maximum_baseline_m: float = 5.0,
    maximum_axis_angle_deg: float = 75.0,
    minimum_similarity: float = 0.65,
    minimum_margin: float = 0.01,
    maximum_epipolar_error_px: float = 2.0,
    epipolar_candidate_topk: int = 1,
    epipolar_recovered_minimum_similarity: float = -1.0,
    epipolar_recovered_minimum_margin: float = -1.0,
    minimum_track_views: int = 3,
    require_cycle: bool = True,
    allow_chain_tracks: bool = False,
    return_pair_sidecar: bool = False,
    precomputed_pairs: list[tuple[int, int]] | None = None,
    precomputed_pair_matches: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    | None = None,
    precomputed_pair_match_diagnostics: dict[
        tuple[int, int], dict[str, int]
    ]
    | None = None,
    precomputed_confidence_includes_detector_scores: bool = False,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]] | tuple[
    dict[str, torch.Tensor], dict[str, float | int], dict
]:
    """Build map-independent native 2D tracks from a local camera graph."""
    query_count = len(descriptors)
    if len(keypoints) != query_count:
        raise ValueError("Descriptor and keypoint camera tables must align")
    camera_K = torch.as_tensor(camera_K, dtype=torch.float64).cpu()
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float64).cpu()
    uses_precomputed_pair_matches = precomputed_pairs is not None
    if uses_precomputed_pair_matches:
        pairs = [tuple(map(int, pair)) for pair in precomputed_pairs]
        if pairs != sorted(set(pairs)) or any(
            left < 0
            or left >= right
            or right >= query_count
            for left, right in pairs
        ):
            raise ValueError(
                "Precomputed pairs must be unique, sorted and canonical"
            )
        if pair_budget is not None and int(pair_budget) != len(pairs):
            raise ValueError("Precomputed pair count differs from the exact budget")
        if precomputed_pair_matches is None:
            raise ValueError("Precomputed pair selection lacks exact matches")
        if set(precomputed_pair_matches) != set(pairs):
            raise ValueError(
                "Precomputed matches must contain every selected pair exactly once"
            )
        if precomputed_pair_match_diagnostics is not None and set(
            precomputed_pair_match_diagnostics
        ) != set(pairs):
            raise ValueError(
                "Precomputed diagnostics must contain every selected pair exactly once"
            )
        if detector_scores is not None and not bool(
            precomputed_confidence_includes_detector_scores
        ):
            raise ValueError(
                "Precomputed match confidence must attest detector-score weighting"
            )
    else:
        if (
            precomputed_pair_matches is not None
            or precomputed_pair_match_diagnostics is not None
            or precomputed_confidence_includes_detector_scores
        ):
            raise ValueError(
                "Precomputed match fields require explicit precomputed pairs"
            )
        pairs = candidate_camera_pairs(
            pose_w2c,
            neighbors=pair_neighbors,
            minimum_baseline_m=minimum_baseline_m,
            maximum_baseline_m=maximum_baseline_m,
            maximum_axis_angle_deg=maximum_axis_angle_deg,
            policy=pair_policy,
            pair_budget=pair_budget,
            camera_K=camera_K,
            image_hw=pair_image_hw,
            scene_points_xyz=pair_scene_points_xyz,
            minimum_overlap_jaccard=pair_minimum_overlap_jaccard,
            minimum_joint_visibility_points=pair_minimum_joint_visibility_points,
            parallax_saturation_deg=pair_parallax_saturation_deg,
            diversity_weight=pair_diversity_weight,
            candidate_pool_per_camera=pair_candidate_pool_per_camera,
            scene_depth_m=pair_scene_depth_m,
            minimum_expected_parallax_deg=(
                pair_minimum_expected_parallax_deg
            ),
            near_fraction=pair_near_fraction,
            maximum_baseline_depth_ratio=(
                pair_maximum_baseline_depth_ratio
            ),
        )
    pair_matches = {}
    pair_match_diagnostics = {}
    raw_match_count = 0
    device = torch.device(device)
    for left, right in pairs:
        if uses_precomputed_pair_matches:
            source, target, confidence = precomputed_pair_matches[(left, right)]
            source = torch.as_tensor(source, dtype=torch.long).cpu().reshape(-1)
            target = torch.as_tensor(target, dtype=torch.long).cpu().reshape(-1)
            confidence = (
                torch.as_tensor(confidence, dtype=torch.float32).cpu().reshape(-1)
            )
            if (
                source.numel() != target.numel()
                or source.numel() != confidence.numel()
            ):
                raise ValueError("Precomputed pair match columns must align")
            if source.numel() and (
                int(source.min()) < 0
                or int(source.max()) >= int(keypoints[left].shape[0])
                or int(target.min()) < 0
                or int(target.max()) >= int(keypoints[right].shape[0])
            ):
                raise ValueError("Precomputed pair keypoint index is out of range")
            if (
                source.unique().numel() != source.numel()
                or target.unique().numel() != target.numel()
            ):
                raise ValueError("Precomputed pair matches are not one-to-one")
            if not bool(torch.isfinite(confidence).all()) or bool(
                (confidence < 0).any()
            ):
                raise ValueError(
                    "Precomputed pair confidence must be finite and non-negative"
                )
            pair_match_diagnostics[(left, right)] = dict(
                (precomputed_pair_match_diagnostics or {}).get((left, right), {})
            )
        else:
            match_result = reciprocal_epipolar_matches(
                descriptors[left].to(device),
                descriptors[right].to(device),
                keypoints[left],
                keypoints[right],
                camera_K[left],
                pose_w2c[left],
                camera_K[right],
                pose_w2c[right],
                minimum_similarity=minimum_similarity,
                minimum_margin=minimum_margin,
                maximum_epipolar_error_px=maximum_epipolar_error_px,
                epipolar_candidate_topk=epipolar_candidate_topk,
                recovered_minimum_similarity=(
                    epipolar_recovered_minimum_similarity
                ),
                recovered_minimum_margin=epipolar_recovered_minimum_margin,
                return_diagnostics=return_pair_sidecar,
            )
            if bool(return_pair_sidecar):
                source, target, confidence, match_diagnostics = match_result
                pair_match_diagnostics[(left, right)] = match_diagnostics
            else:
                source, target, confidence = match_result
        if source.numel() == 0:
            continue
        raw_match_count += int(source.numel())
        if detector_scores is not None and not uses_precomputed_pair_matches:
            confidence = confidence * torch.sqrt(
                detector_scores[left][source].float().clamp_min(0.0)
                * detector_scores[right][target].float().clamp_min(0.0)
            )
        pair_matches[(left, right)] = (source, target, confidence)
    keypoint_counts = [int(value.shape[0]) for value in keypoints]
    cycle_support = (
        _cycle_supported_pair_edges(pair_matches, keypoint_counts)
        if require_cycle
        else {
            pair: torch.ones(match[0].numel(), dtype=torch.bool)
            for pair, match in pair_matches.items()
        }
    )
    keypoint_offsets = [0]
    for count in keypoint_counts:
        keypoint_offsets.append(keypoint_offsets[-1] + count)
    graded_diagnostics = {
        "track_graded_cycle_edge_count": 0,
        "track_graded_chain_edge_count": 0,
        "track_graded_conflict_rejected_edge_count": 0,
    }
    component_cycle_seeded = {}
    pair_edge_status = {
        "accepted": {
            pair: cycle_support[pair].clone() for pair in pair_matches
        },
        "conflict_rejected": {
            pair: torch.zeros(match[0].numel(), dtype=torch.bool)
            for pair, match in pair_matches.items()
        },
    }
    if allow_chain_tracks:
        if not require_cycle:
            raise ValueError(
                "allow_chain_tracks requires cycle support for graded tracks"
            )
        (
            components,
            node_confidence,
            component_cycle_seeded,
            graded_diagnostics,
            pair_edge_status,
        ) = _graded_track_components(
            pair_matches, cycle_support, keypoint_offsets
        )
        supported_edge_count = (
            graded_diagnostics["track_graded_cycle_edge_count"]
            + graded_diagnostics["track_graded_chain_edge_count"]
        )
        active_nodes = {
            node for nodes in components.values() for node in nodes
        }
    else:
        disjoint = _SparseDisjointSet()
        node_confidence = defaultdict(float)
        supported_edge_count = 0
        for pair, (source, target, confidence) in pair_matches.items():
            keep = cycle_support[pair]
            left, right = pair
            for source_index, target_index, edge_confidence in zip(
                source[keep].tolist(),
                target[keep].tolist(),
                confidence[keep].tolist(),
            ):
                source_node = keypoint_offsets[left] + source_index
                target_node = keypoint_offsets[right] + target_index
                disjoint.union(source_node, target_node)
                node_confidence[source_node] = max(
                    node_confidence[source_node], float(edge_confidence)
                )
                node_confidence[target_node] = max(
                    node_confidence[target_node], float(edge_confidence)
                )
                supported_edge_count += 1
        components = defaultdict(list)
        for node in disjoint.parent:
            components[disjoint.find(node)].append(node)
        active_nodes = set(disjoint.parent)
        component_cycle_seeded = {
            root: bool(require_cycle) for root in components
        }
    node_query = {}
    for query, (start, end) in enumerate(
        zip(keypoint_offsets[:-1], keypoint_offsets[1:])
    ):
        for node in range(start, end):
            if node in active_nodes:
                node_query[node] = (query, node - start)
    track_indices = []
    query_indices = []
    keypoint_indices = []
    confidences = []
    track_levels = []
    track_count = 0
    level_a_track_count = 0
    level_b_track_count = 0
    rejected_duplicate_query = 0
    node_to_track = {}
    for root, nodes in components.items():
        observations = [node_query[node] for node in nodes]
        queries = [item[0] for item in observations]
        if len(set(queries)) != len(queries):
            rejected_duplicate_query += 1
            continue
        if len(queries) < int(minimum_track_views):
            continue
        for node, (query, keypoint) in zip(nodes, observations):
            track_indices.append(track_count)
            query_indices.append(query)
            keypoint_indices.append(keypoint)
            confidences.append(node_confidence[node])
            node_to_track[node] = track_count
        level = 2 if component_cycle_seeded.get(root, False) else 1
        track_levels.append(level)
        if level == 2:
            level_a_track_count += 1
        else:
            level_b_track_count += 1
        track_count += 1
    tracks = {
        "track_index": torch.as_tensor(track_indices, dtype=torch.long),
        "query_index": torch.as_tensor(query_indices, dtype=torch.long),
        "keypoint_index": torch.as_tensor(keypoint_indices, dtype=torch.long),
        "confidence": torch.as_tensor(confidences, dtype=torch.float32),
        "track_level": torch.as_tensor(track_levels, dtype=torch.int8),
    }
    diagnostics = {
        "track_camera_pair_candidate_count": len(pairs),
        "track_camera_pair_matched_count": len(pair_matches),
        "track_raw_reciprocal_epipolar_edge_count": raw_match_count,
        "track_epipolar_candidate_topk": int(epipolar_candidate_topk),
        "track_epipolar_recovered_minimum_similarity": float(
            epipolar_recovered_minimum_similarity
        ),
        "track_epipolar_recovered_minimum_margin": float(
            epipolar_recovered_minimum_margin
        ),
        "track_cycle_supported_edge_count": supported_edge_count,
        "track_count": track_count,
        "track_level_a_count": level_a_track_count,
        "track_level_b_count": level_b_track_count,
        "track_allow_chain_tracks": int(bool(allow_chain_tracks)),
        "track_camera_pair_policy": str(pair_policy),
        "track_camera_pair_budget": int(len(pairs)),
        "track_pair_matches_reused": int(uses_precomputed_pair_matches),
        "track_observation_count": len(track_indices),
        "track_rejected_duplicate_query_component_count": (
            rejected_duplicate_query
        ),
        **graded_diagnostics,
    }
    if not bool(return_pair_sidecar):
        return tracks, diagnostics

    geometry_table = _camera_pair_geometry_table(
        pairs,
        pose_w2c,
        camera_K=camera_K,
        image_hw=pair_image_hw,
        scene_points_xyz=pair_scene_points_xyz,
    )
    diagnostic_names = sorted(
        {
            name
            for diagnostic in pair_match_diagnostics.values()
            for name in diagnostic
        }
    )
    match_columns = {
        name: torch.zeros(len(pairs), dtype=torch.long)
        for name in diagnostic_names
    }
    cycle_supported_count = torch.zeros(len(pairs), dtype=torch.long)
    graph_accepted_count = torch.zeros(len(pairs), dtype=torch.long)
    conflict_rejected_count = torch.zeros(len(pairs), dtype=torch.long)
    final_component_edge_count = torch.zeros(len(pairs), dtype=torch.long)
    final_track_offsets = [0]
    final_track_indices = []
    for pair_index, pair in enumerate(pairs):
        for name, value in pair_match_diagnostics.get(pair, {}).items():
            match_columns[name][pair_index] = int(value)
        if pair not in pair_matches:
            final_track_offsets.append(final_track_offsets[-1])
            continue
        source, target, _ = pair_matches[pair]
        cycle_supported_count[pair_index] = int(cycle_support[pair].sum())
        accepted = pair_edge_status["accepted"][pair]
        conflict_rejected_count[pair_index] = int(
            pair_edge_status["conflict_rejected"][pair].sum()
        )
        graph_accepted_count[pair_index] = int(accepted.sum())
        left, right = pair
        contributed_tracks = []
        for local_edge in torch.nonzero(accepted, as_tuple=False).reshape(-1).tolist():
            source_node = keypoint_offsets[left] + int(source[local_edge])
            target_node = keypoint_offsets[right] + int(target[local_edge])
            source_track = node_to_track.get(source_node, -1)
            target_track = node_to_track.get(target_node, -1)
            if source_track < 0 or source_track != target_track:
                continue
            final_component_edge_count[pair_index] += 1
            contributed_tracks.append(source_track)
        unique_tracks = sorted(set(contributed_tracks))
        final_track_indices.extend(unique_tracks)
        final_track_offsets.append(final_track_offsets[-1] + len(unique_tracks))
    pair_sidecar = {
        "schema": "lafgs_mapping_track_pair_sidecar",
        "version": 1,
        "policy": {
            "name": str(pair_policy),
            "neighbors": int(pair_neighbors),
            "exact_pair_budget": int(len(pairs)),
            "minimum_baseline_m": float(minimum_baseline_m),
            "maximum_baseline_m": float(maximum_baseline_m),
            "maximum_axis_angle_deg": float(maximum_axis_angle_deg),
            "minimum_overlap_jaccard": float(pair_minimum_overlap_jaccard),
            "minimum_joint_visibility_points": int(
                pair_minimum_joint_visibility_points
            ),
            "parallax_saturation_deg": float(pair_parallax_saturation_deg),
            "diversity_weight": float(pair_diversity_weight),
            "candidate_pool_per_camera": int(pair_candidate_pool_per_camera),
            "uses_descriptors_for_selection": False,
            "uses_precomputed_pair_matches": bool(uses_precomputed_pair_matches),
            "uses_test_queries": False,
            "overlap_constraint_applied": str(pair_policy)
            == "parallax_diverse",
        },
        "pair": {
            **geometry_table,
            **match_columns,
            "raw_match_count": match_columns.get(
                "raw_top1_reciprocal_count",
                torch.zeros(len(pairs), dtype=torch.long),
            ),
            "accepted_match_count": match_columns.get(
                "final_reciprocal_epipolar_count",
                torch.zeros(len(pairs), dtype=torch.long),
            ),
            "rejected_ambiguity_count": match_columns.get(
                "ambiguity_rejected_count",
                torch.zeros(len(pairs), dtype=torch.long),
            ),
            "rejected_epipolar_count": match_columns.get(
                "epipolar_rejected_after_descriptor_count",
                torch.zeros(len(pairs), dtype=torch.long),
            ),
            "cycle_supported_edge_count": cycle_supported_count,
            "cycle_supported_match_count": cycle_supported_count,
            "graph_accepted_edge_count": graph_accepted_count,
            "conflict_rejected_edge_count": conflict_rejected_count,
            "final_component_edge_count": final_component_edge_count,
            "final_track_offsets": torch.as_tensor(
                final_track_offsets, dtype=torch.long
            ),
            "final_track_indices": torch.as_tensor(
                final_track_indices, dtype=torch.long
            ),
            "triangulated_track_count": torch.full(
                (len(pairs),), -1, dtype=torch.long
            ),
            "actual_triangulation_parallax_median_deg": torch.full(
                (len(pairs),), float("nan"), dtype=torch.float64
            ),
        },
        "count_semantics": {
            "raw_top1_reciprocal_count": (
                "Ungated descriptor Top-1 mutual-nearest-neighbour edges."
            ),
            "raw_match_count": "Alias of raw_top1_reciprocal_count.",
            "descriptor_accepted_before_epipolar_count": (
                "Raw Top-1 reciprocal edges passing similarity and both margins."
            ),
            "epipolar_accepted_top1_count": (
                "Those descriptor-accepted raw Top-1 edges passing the known-pose "
                "epipolar threshold."
            ),
            "final_reciprocal_epipolar_count": (
                "Edges emitted by the configured matcher; with top-K>1 this may "
                "include explicitly counted epipolar recoveries."
            ),
            "accepted_match_count": (
                "Alias of final_reciprocal_epipolar_count."
            ),
            "rejected_ambiguity_count": "Alias of ambiguity_rejected_count.",
            "rejected_epipolar_count": (
                "Alias of epipolar_rejected_after_descriptor_count."
            ),
            "cycle_supported_edge_count": (
                "Emitted edges participating in an exact three-camera keypoint cycle."
            ),
            "cycle_supported_match_count": (
                "Alias of cycle_supported_edge_count."
            ),
            "conflict_rejected_edge_count": (
                "Emitted edges rejected because union would duplicate a camera in "
                "one Track component."
            ),
            "final_component_edge_count": (
                "Accepted direct pair edges retained in a minimum-view final Track."
            ),
        },
    }
    if str(pair_policy) == "parallax_stratified":
        pair_sidecar["policy"].update(
            {
                "minimum_expected_parallax_deg": float(
                    pair_minimum_expected_parallax_deg
                ),
                "near_fraction": float(pair_near_fraction),
                "maximum_baseline_depth_ratio": float(
                    pair_maximum_baseline_depth_ratio
                ),
                "scene_depth_estimator": (
                    "median_positive_mapping_keypoint_depth"
                ),
            }
        )
    return tracks, diagnostics, pair_sidecar


def attach_pair_triangulation_statistics(
    pair_sidecar: dict,
    tracks: dict[str, torch.Tensor],
    track_geometry: dict[str, torch.Tensor],
    pose_w2c: torch.Tensor,
) -> dict:
    """Attach exact per-pair statistics from the completed triangulation.

    Only final Tracks that contain an accepted direct edge from the pair are
    considered.  The reported angle is recomputed from the final 3D point and
    the two camera centers; it is not copied from a pose-only proxy.
    """
    if pair_sidecar.get("schema") != "lafgs_mapping_track_pair_sidecar":
        raise ValueError("Unexpected Track pair sidecar schema")
    pair = pair_sidecar["pair"]
    left = torch.as_tensor(pair["left_query_index"], dtype=torch.long)
    right = torch.as_tensor(pair["right_query_index"], dtype=torch.long)
    offsets = torch.as_tensor(pair["final_track_offsets"], dtype=torch.long)
    indices = torch.as_tensor(pair["final_track_indices"], dtype=torch.long)
    if offsets.numel() != left.numel() + 1 or int(offsets[-1]) != indices.numel():
        raise ValueError("Malformed pair-to-final-Track CSR")
    geometry_xyz = torch.as_tensor(
        track_geometry["triangulated_xyz"], dtype=torch.float64
    )
    triangulated = torch.as_tensor(
        track_geometry["triangulated"], dtype=torch.bool
    )
    track_count = int(torch.as_tensor(tracks["track_level"]).numel())
    if geometry_xyz.shape[0] != track_count or triangulated.numel() != track_count:
        raise ValueError("Track table and triangulation table do not align")
    centers, _ = _camera_centers_and_axes(pose_w2c)
    triangulated_count = torch.zeros(left.numel(), dtype=torch.long)
    actual_parallax = torch.full(
        (left.numel(),), float("nan"), dtype=torch.float64
    )
    for pair_index in range(int(left.numel())):
        begin = int(offsets[pair_index])
        end = int(offsets[pair_index + 1])
        pair_tracks = indices[begin:end]
        if pair_tracks.numel() == 0:
            continue
        pair_tracks = pair_tracks[triangulated[pair_tracks]]
        if pair_tracks.numel() == 0:
            continue
        xyz = geometry_xyz[pair_tracks]
        finite = torch.isfinite(xyz).all(dim=1)
        xyz = xyz[finite]
        if xyz.numel() == 0:
            continue
        triangulated_count[pair_index] = int(xyz.shape[0])
        ray_left = F.normalize(xyz - centers[left[pair_index]], dim=1)
        ray_right = F.normalize(xyz - centers[right[pair_index]], dim=1)
        cosine = (ray_left * ray_right).sum(dim=1).clamp(-1.0, 1.0)
        actual_parallax[pair_index] = torch.rad2deg(torch.acos(cosine)).median()
    pair["triangulated_track_count"] = triangulated_count
    pair["actual_triangulation_parallax_median_deg"] = actual_parallax
    pair_sidecar["triangulation_attached"] = True
    pair_sidecar["actual_triangulation_parallax_semantics"] = (
        "Median exact ray angle over triangulated final Tracks containing an "
        "accepted direct edge from this camera pair."
    )
    return pair_sidecar


def _deduplicate_landmark_query(
    landmark_index: torch.Tensor,
    query_index: torch.Tensor,
    confidence: torch.Tensor,
    query_count: int,
) -> torch.Tensor:
    """Keep the highest-confidence observation for each landmark/query pair."""
    key = landmark_index.long() * int(query_count) + query_index.long()
    confidence_order = torch.argsort(
        confidence, descending=True, stable=True
    )
    key_order = torch.argsort(key[confidence_order], stable=True)
    ordered = confidence_order[key_order]
    ordered_key = key[ordered]
    keep = torch.ones(ordered.numel(), dtype=torch.bool)
    if ordered.numel() > 1:
        keep[1:] = ordered_key[1:] != ordered_key[:-1]
    return ordered[keep]


def _camera_rays(
    uv: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    uv = uv.to(dtype=torch.float64)
    K = K.to(dtype=torch.float64)
    pose_w2c = pose_w2c.to(dtype=torch.float64)
    homogeneous = torch.cat(
        (uv, torch.ones((uv.shape[0], 1), dtype=uv.dtype)), dim=1
    )
    direction_camera = torch.linalg.solve(K, homogeneous[:, :, None]).squeeze(2)
    rotation = pose_w2c[:, :3, :3]
    translation = pose_w2c[:, :3, 3]
    center = -torch.einsum("nji,nj->ni", rotation, translation)
    direction = torch.einsum("nji,nj->ni", rotation, direction_camera)
    direction = torch.nn.functional.normalize(direction, dim=1)
    return center, direction


def _weighted_ray_intersection(
    center: torch.Tensor,
    direction: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(3, dtype=center.dtype)
    projector = identity[None] - direction[:, :, None] * direction[:, None, :]
    weighted = projector * weight[:, None, None]
    normal = weighted.sum(dim=0)
    rhs = torch.einsum("nij,nj->i", weighted, center)
    eigenvalues = torch.linalg.eigvalsh(normal)
    if (
        not bool(torch.isfinite(eigenvalues).all())
        or float(eigenvalues[0]) <= 1e-12
    ):
        raise torch.linalg.LinAlgError("Degenerate ray intersection")
    point = torch.linalg.solve(normal, rhs)
    condition = eigenvalues[-1] / eigenvalues[0]
    return point, normal, condition


def _project(
    point: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    point_h = torch.cat((point, point.new_ones(1)))
    camera = torch.einsum("nij,j->ni", pose_w2c, point_h)[:, :3]
    depth = camera[:, 2]
    projected_h = torch.einsum("nij,nj->ni", K, camera)
    uv = projected_h[:, :2] / projected_h[:, 2:3].clamp_min(1e-12)
    return uv, depth


def _reprojection_normal_matrix(
    point: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Build J.T W J in metric world coordinates for pixel residuals."""
    point_h = torch.cat((point, point.new_ones(1)))
    camera = torch.einsum("nij,j->ni", pose_w2c, point_h)[:, :3]
    projected_h = torch.einsum("nij,nj->ni", K, camera)
    denominator = projected_h[:, 2].clamp_min(1e-12)
    jacobian_camera_u = (
        K[:, 0, :] * denominator[:, None]
        - projected_h[:, 0, None] * K[:, 2, :]
    ) / denominator[:, None].square()
    jacobian_camera_v = (
        K[:, 1, :] * denominator[:, None]
        - projected_h[:, 1, None] * K[:, 2, :]
    ) / denominator[:, None].square()
    jacobian_camera = torch.stack(
        (jacobian_camera_u, jacobian_camera_v), dim=1
    )
    jacobian_world = torch.einsum(
        "nij,njk->nik", jacobian_camera, pose_w2c[:, :3, :3]
    )
    return torch.einsum(
        "n,nji,njk->ik", weight, jacobian_world, jacobian_world
    )


def _surface_supported_weak_axis_update(
    *,
    point: torch.Tensor,
    reprojection_normal: torch.Tensor,
    reprojection_covariance: torch.Tensor,
    uv: torch.Tensor,
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    rendered_depth: torch.Tensor,
    base_weight: torch.Tensor,
    huber_m: float,
    maximum_correction_m: float,
    maximum_weak_information_ratio: float,
    minimum_depth_improvement_fraction: float,
    maximum_reprojection_increase_px: float,
    covariance_sigma_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict] | None:
    """Use a frozen rendered surface only along triangulation's weak axis.

    Real-image tracks retain ownership of identity and tangent geometry.  The
    rendered depth is treated as one correlated, noisy measurement and must
    pass an observation-level cross-fit before it can reduce depth uncertainty.
    """
    valid = torch.isfinite(rendered_depth) & (rendered_depth > 0)
    if int(valid.sum()) < 3:
        return None
    eigenvalues, eigenvectors = torch.linalg.eigh(reprojection_normal)
    if (
        not bool(torch.isfinite(eigenvalues).all())
        or float(eigenvalues[1]) <= 1e-12
    ):
        return None
    weak_ratio = float(eigenvalues[0] / eigenvalues[1])
    if weak_ratio > float(maximum_weak_information_ratio):
        return None
    weak_axis = eigenvectors[:, 0]
    depth_jacobian = pose_w2c[:, 2, :3] @ weak_axis
    valid &= depth_jacobian.abs() >= 1e-3
    valid_rows = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_rows.numel() < 3:
        return None

    # Interleaving observations prevents a correction fitted to one local
    # trajectory fragment from validating itself on the same rendered samples.
    fit_rows = valid_rows[::2]
    validation_rows = valid_rows[1::2]
    if fit_rows.numel() < 2 or validation_rows.numel() < 1:
        return None
    projected, camera_depth = _project(point, camera_K, pose_w2c)
    base_depth_residual = camera_depth - rendered_depth
    base_reprojection = torch.linalg.norm(projected - uv, dim=1)

    def solve_delta(rows: torch.Tensor) -> torch.Tensor:
        coefficient = depth_jacobian[rows]
        residual = base_depth_residual[rows]
        weights = base_weight[rows].clamp_min(1e-6)
        delta = -(weights * coefficient * residual).sum() / (
            weights * coefficient.square()
        ).sum().clamp_min(1e-12)
        for _ in range(3):
            revised = residual + coefficient * delta
            robust = torch.where(
                revised.abs() <= float(huber_m),
                torch.ones_like(revised),
                float(huber_m) / revised.abs().clamp_min(1e-12),
            )
            combined = weights * robust
            delta = -(combined * coefficient * residual).sum() / (
                combined * coefficient.square()
            ).sum().clamp_min(1e-12)
        return delta.clamp(
            -float(maximum_correction_m), float(maximum_correction_m)
        )

    fitted_delta = solve_delta(fit_rows)
    candidate = point + fitted_delta * weak_axis
    candidate_projected, candidate_depth = _project(
        candidate, camera_K, pose_w2c
    )
    candidate_reprojection = torch.linalg.norm(candidate_projected - uv, dim=1)
    base_validation_depth = base_depth_residual[validation_rows].abs().median()
    candidate_validation_depth = (
        candidate_depth[validation_rows] - rendered_depth[validation_rows]
    ).abs().median()
    depth_improved = candidate_validation_depth <= base_validation_depth * (
        1.0 - float(minimum_depth_improvement_fraction)
    )
    already_consistent = (
        base_validation_depth <= float(covariance_sigma_m)
        and candidate_validation_depth <= float(covariance_sigma_m)
    )
    reprojection_safe = candidate_reprojection[validation_rows].median() <= (
        base_reprojection[validation_rows].median()
        + float(maximum_reprojection_increase_px)
    )
    if not bool((depth_improved | already_consistent) & reprojection_safe):
        return None

    # Refit after the held-out gate.  Surface observations are normalized to
    # one effective measurement because renderer errors are correlated.
    final_delta = solve_delta(valid_rows)
    revised_point = point + final_delta * weak_axis
    revised_projected, revised_depth = _project(
        revised_point, camera_K, pose_w2c
    )
    revised_reprojection = torch.linalg.norm(revised_projected - uv, dim=1)
    if revised_reprojection.median() > (
        base_reprojection.median() + float(maximum_reprojection_increase_px)
    ):
        return None
    revised_depth_residual = revised_depth[valid_rows] - rendered_depth[valid_rows]
    # A capped step that merely moves toward a distant rendered surface must
    # not receive surface-derived covariance.  It remains a valid pure-image
    # triangulation and falls back unchanged.
    if revised_depth_residual.abs().median() > float(covariance_sigma_m):
        return None
    if revised_depth_residual.abs().median() > (
        base_depth_residual[valid_rows].abs().median()
        + 0.25 * float(covariance_sigma_m)
    ):
        return None

    surface_robust = torch.where(
        revised_depth_residual.abs() <= float(huber_m),
        torch.ones_like(revised_depth_residual),
        float(huber_m) / revised_depth_residual.abs().clamp_min(1e-12),
    )
    surface_weight = base_weight[valid_rows] * surface_robust
    surface_weight /= surface_weight.sum().clamp_min(1e-12)
    effective_jacobian_sq = (
        surface_weight * depth_jacobian[valid_rows].square()
    ).sum()
    surface_precision = effective_jacobian_sq / max(
        float(covariance_sigma_m) ** 2, 1e-12
    )
    fused_precision = torch.linalg.inv(reprojection_covariance) + (
        surface_precision * weak_axis[:, None] * weak_axis[None, :]
    )
    fused_covariance = torch.linalg.inv(fused_precision)
    fused_eigenvalues = torch.linalg.eigvalsh(fused_precision)
    fused_condition = fused_eigenvalues[-1] / fused_eigenvalues[0].clamp_min(1e-12)
    report = {
        "correction_m": float(final_delta.abs()),
        "signed_correction_m": float(final_delta),
        "weak_information_ratio": weak_ratio,
        "depth_before_m": float(base_depth_residual[valid_rows].abs().median()),
        "depth_after_m": float(revised_depth_residual.abs().median()),
        "validation_depth_before_m": float(base_validation_depth),
        "validation_depth_after_m": float(candidate_validation_depth),
        "reprojection_delta_px": float(
            revised_reprojection.median() - base_reprojection.median()
        ),
        "observation_count": int(valid_rows.numel()),
    }
    return revised_point, fused_covariance, fused_condition, report


def robust_triangulate_associations(
    *,
    landmark_count: int,
    landmark_index: torch.Tensor,
    query_index: torch.Tensor,
    uv: torch.Tensor,
    confidence: torch.Tensor,
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    query_bin: torch.Tensor | None = None,
    rendered_depth: torch.Tensor | None = None,
    maximum_observations_per_landmark: int = 32,
    minimum_views: int = 3,
    minimum_view_bins: int = 2,
    huber_delta_px: float = 2.0,
    iterations: int = 3,
    minimum_parallax_deg: float = 1.0,
    parallax_quantile: float = 0.75,
    maximum_reprojection_px: float = 2.0,
    maximum_condition_number: float = 1e6,
    maximum_covariance_trace_m2: float = float("inf"),
    maximum_rendered_depth_residual_m: float = float("inf"),
    minimum_rendered_depth_observations: int = 0,
    surface_support_enabled: bool = False,
    surface_support_huber_m: float = 0.02,
    surface_support_maximum_correction_m: float = 0.08,
    surface_support_maximum_weak_information_ratio: float = 0.25,
    surface_support_minimum_depth_improvement_fraction: float = 0.10,
    surface_support_maximum_reprojection_increase_px: float = 0.05,
    surface_support_covariance_sigma_m: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Robustly triangulate descriptor-only cross-view landmark associations."""
    previous_thread_count = torch.get_num_threads()
    # Thousands of independent 3x3 decompositions are substantially slower
    # when every call fans out across the host's full OpenMP thread pool.
    torch.set_num_threads(1)
    landmark_count = int(landmark_count)
    landmark_index = torch.as_tensor(landmark_index, dtype=torch.long).cpu()
    query_index = torch.as_tensor(query_index, dtype=torch.long).cpu()
    uv = torch.as_tensor(uv, dtype=torch.float64).cpu()
    confidence = torch.as_tensor(confidence, dtype=torch.float64).cpu()
    camera_K = torch.as_tensor(camera_K, dtype=torch.float64).cpu()
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float64).cpu()
    if query_bin is None:
        query_bin = camera_pose_bins(pose_w2c, 8)
    query_bin = torch.as_tensor(query_bin, dtype=torch.long).cpu()
    if rendered_depth is not None:
        rendered_depth = torch.as_tensor(
            rendered_depth, dtype=torch.float64
        ).cpu()
    if not (
        landmark_index.numel()
        == query_index.numel()
        == uv.shape[0]
        == confidence.numel()
    ):
        raise ValueError("Association tensors must have the same leading size")
    if landmark_index.numel() == 0:
        raise ValueError("At least one association is required")
    if int(query_index.max()) >= int(camera_K.shape[0]):
        raise ValueError("query_index exceeds the supplied camera table")

    keep = _deduplicate_landmark_query(
        landmark_index, query_index, confidence, int(camera_K.shape[0])
    )
    landmark_index = landmark_index[keep]
    query_index = query_index[keep]
    uv = uv[keep]
    confidence = confidence[keep]
    if rendered_depth is not None:
        rendered_depth = rendered_depth[keep]
    order = torch.argsort(landmark_index, stable=True)
    landmark_index = landmark_index[order]
    query_index = query_index[order]
    uv = uv[order]
    confidence = confidence[order]
    if rendered_depth is not None:
        rendered_depth = rendered_depth[order]

    triangulated_xyz = torch.full(
        (landmark_count, 3), float("nan"), dtype=torch.float64
    )
    observation_count = torch.zeros(landmark_count, dtype=torch.long)
    distinct_view_count = torch.zeros(landmark_count, dtype=torch.long)
    distinct_view_bin_count = torch.zeros(landmark_count, dtype=torch.long)
    reprojection_median_px = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    reprojection_p90_px = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    parallax_deg = torch.zeros(landmark_count, dtype=torch.float64)
    condition_number = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    covariance_trace = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    covariance_matrix = torch.full(
        (landmark_count, 3, 3), float("nan"), dtype=torch.float64
    )
    image_only_xyz = torch.full(
        (landmark_count, 3), float("nan"), dtype=torch.float64
    )
    image_only_covariance_trace = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    image_only_covariance_matrix = torch.full(
        (landmark_count, 3, 3), float("nan"), dtype=torch.float64
    )
    image_only_reprojection_median_px = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    image_only_reprojection_p90_px = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    rendered_depth_signed_median_m = torch.full(
        (landmark_count,), float("nan"), dtype=torch.float64
    )
    rendered_depth_absolute_median_m = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    rendered_depth_observation_count = torch.zeros(
        landmark_count, dtype=torch.long
    )
    triangulated = torch.zeros(landmark_count, dtype=torch.bool)
    surface_supported = torch.zeros(landmark_count, dtype=torch.bool)
    surface_correction_m = torch.zeros(landmark_count, dtype=torch.float64)
    surface_weak_information_ratio = torch.full(
        (landmark_count,), float("nan"), dtype=torch.float64
    )
    surface_depth_improvement_m = torch.zeros(
        landmark_count, dtype=torch.float64
    )
    surface_reprojection_delta_px = torch.zeros(
        landmark_count, dtype=torch.float64
    )

    unique_landmarks, counts = torch.unique_consecutive(
        landmark_index, return_counts=True
    )
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    for group, landmark in enumerate(unique_landmarks.tolist()):
        start = int(offsets[group])
        end = int(offsets[group + 1])
        selected = torch.arange(start, end)
        maximum = int(maximum_observations_per_landmark)
        if maximum > 0 and selected.numel() > maximum:
            rank = torch.argsort(
                confidence[selected], descending=True, stable=True
            )
            selected = selected[rank[:maximum]]
        queries = query_index[selected]
        if queries.numel() < int(minimum_views):
            continue
        bins = torch.unique(query_bin[queries])
        distinct_view_count[landmark] = int(queries.numel())
        distinct_view_bin_count[landmark] = int(bins.numel())
        if bins.numel() < int(minimum_view_bins):
            continue
        centers, directions = _camera_rays(
            uv[selected], camera_K[queries], pose_w2c[queries]
        )
        base_weight = confidence[selected].clamp_min(1e-4)
        base_weight /= base_weight.mean().clamp_min(1e-8)
        weight = base_weight
        try:
            point, normal, condition = _weighted_ray_intersection(
                centers, directions, weight
            )
            for _ in range(max(int(iterations), 1)):
                projected, depth = _project(
                    point, camera_K[queries], pose_w2c[queries]
                )
                residual = torch.linalg.norm(projected - uv[selected], dim=1)
                robust = torch.where(
                    residual <= float(huber_delta_px),
                    torch.ones_like(residual),
                    float(huber_delta_px) / residual.clamp_min(1e-8),
                )
                robust = robust * (depth > 0).to(robust.dtype)
                point, normal, condition = _weighted_ray_intersection(
                    centers, directions, base_weight * robust
                )
        except (RuntimeError, torch.linalg.LinAlgError):
            continue
        projected, depth = _project(
            point, camera_K[queries], pose_w2c[queries]
        )
        residual = torch.linalg.norm(projected - uv[selected], dim=1)
        finite = torch.isfinite(residual) & (depth > 0)
        if int(finite.sum()) < int(minimum_views):
            continue
        robust_inlier = finite & (
            residual
            <= max(
                float(maximum_reprojection_px) * 2.0,
                float(huber_delta_px) * 2.0,
            )
        )
        if int(robust_inlier.sum()) < int(minimum_views):
            robust_inlier = finite
        inlier_directions = directions[robust_inlier]
        pair_cosine = (inlier_directions @ inlier_directions.T).clamp(
            -1.0, 1.0
        )
        upper = torch.triu_indices(
            pair_cosine.shape[0], pair_cosine.shape[1], offset=1
        )
        pair_angles = torch.rad2deg(
            torch.acos(pair_cosine[upper[0], upper[1]])
        )
        if pair_angles.numel() > 0:
            parallax_deg[landmark] = torch.quantile(
                pair_angles,
                min(max(float(parallax_quantile), 0.0), 1.0),
            )
        residual = residual[finite]
        sigma2 = residual.square().median().clamp_min(1e-8)
        final_projected, final_depth = _project(
            point, camera_K[queries], pose_w2c[queries]
        )
        final_residual = torch.linalg.norm(
            final_projected - uv[selected], dim=1
        )
        final_robust = torch.where(
            final_residual <= float(huber_delta_px),
            torch.ones_like(final_residual),
            float(huber_delta_px) / final_residual.clamp_min(1e-8),
        )
        final_weight = (
            base_weight
            * final_robust
            * (final_depth > 0).to(final_robust.dtype)
        )
        reprojection_normal = _reprojection_normal_matrix(
            point,
            camera_K[queries],
            pose_w2c[queries],
            final_weight,
        )
        reprojection_eigenvalues = torch.linalg.eigvalsh(
            reprojection_normal
        )
        if (
            not bool(torch.isfinite(reprojection_eigenvalues).all())
            or float(reprojection_eigenvalues[0]) <= 1e-12
        ):
            continue
        covariance = torch.linalg.inv(reprojection_normal) * sigma2
        condition = (
            reprojection_eigenvalues[-1] / reprojection_eigenvalues[0]
        )
        image_only_xyz[landmark] = point
        image_only_covariance_trace[landmark] = torch.trace(covariance)
        image_only_covariance_matrix[landmark] = covariance
        image_only_reprojection_median_px[landmark] = residual.median()
        image_only_reprojection_p90_px[landmark] = torch.quantile(residual, 0.9)
        surface_report = None
        if bool(surface_support_enabled) and rendered_depth is not None:
            surface_result = _surface_supported_weak_axis_update(
                point=point,
                reprojection_normal=reprojection_normal,
                reprojection_covariance=covariance,
                uv=uv[selected],
                camera_K=camera_K[queries],
                pose_w2c=pose_w2c[queries],
                rendered_depth=rendered_depth[selected],
                base_weight=base_weight,
                huber_m=surface_support_huber_m,
                maximum_correction_m=surface_support_maximum_correction_m,
                maximum_weak_information_ratio=(
                    surface_support_maximum_weak_information_ratio
                ),
                minimum_depth_improvement_fraction=(
                    surface_support_minimum_depth_improvement_fraction
                ),
                maximum_reprojection_increase_px=(
                    surface_support_maximum_reprojection_increase_px
                ),
                covariance_sigma_m=surface_support_covariance_sigma_m,
            )
            if surface_result is not None:
                point, covariance, condition, surface_report = surface_result
                projected, depth = _project(
                    point, camera_K[queries], pose_w2c[queries]
                )
                residual = torch.linalg.norm(projected - uv[selected], dim=1)
                finite = torch.isfinite(residual) & (depth > 0)
        triangulated_xyz[landmark] = point
        observation_count[landmark] = int(finite.sum())
        reprojection_median_px[landmark] = residual.median()
        reprojection_p90_px[landmark] = torch.quantile(residual, 0.9)
        condition_number[landmark] = condition
        covariance_trace[landmark] = torch.trace(covariance)
        covariance_matrix[landmark] = covariance
        if surface_report is not None:
            surface_supported[landmark] = True
            surface_correction_m[landmark] = surface_report["correction_m"]
            surface_weak_information_ratio[landmark] = surface_report[
                "weak_information_ratio"
            ]
            surface_depth_improvement_m[landmark] = (
                surface_report["depth_before_m"]
                - surface_report["depth_after_m"]
            )
            surface_reprojection_delta_px[landmark] = surface_report[
                "reprojection_delta_px"
            ]
        if rendered_depth is not None:
            valid_depth = (
                finite
                & torch.isfinite(rendered_depth[selected])
                & (rendered_depth[selected] > 0)
            )
            if bool(valid_depth.any()):
                depth_delta = (
                    depth[valid_depth] - rendered_depth[selected][valid_depth]
                )
                rendered_depth_signed_median_m[landmark] = depth_delta.median()
                rendered_depth_absolute_median_m[landmark] = (
                    depth_delta.abs().median()
                )
                rendered_depth_observation_count[landmark] = int(
                    valid_depth.sum()
                )
        triangulated[landmark] = True

    covariance_valid = covariance_trace <= float(maximum_covariance_trace_m2)
    depth_required = int(minimum_rendered_depth_observations) > 0
    depth_valid = (
        (
            rendered_depth_observation_count
            >= int(minimum_rendered_depth_observations)
        )
        & (
            rendered_depth_absolute_median_m
            <= float(maximum_rendered_depth_residual_m)
        )
        if depth_required
        else torch.ones(landmark_count, dtype=torch.bool)
    )
    high_confidence = (
        triangulated
        & (observation_count >= int(minimum_views))
        & (distinct_view_bin_count >= int(minimum_view_bins))
        & (parallax_deg >= float(minimum_parallax_deg))
        & (reprojection_median_px <= float(maximum_reprojection_px))
        & (condition_number <= float(maximum_condition_number))
        & covariance_valid
        & depth_valid
    )
    result = {
        "triangulated_xyz": triangulated_xyz.float(),
        "triangulated": triangulated,
        "triangulation_high_confidence": high_confidence,
        "triangulation_observation_count": observation_count,
        "triangulation_distinct_view_count": distinct_view_count,
        "triangulation_distinct_view_bin_count": distinct_view_bin_count,
        "triangulation_reprojection_median_px": reprojection_median_px.float(),
        "triangulation_reprojection_p90_px": reprojection_p90_px.float(),
        "triangulation_parallax_deg": parallax_deg.float(),
        "triangulation_condition_number": condition_number.float(),
        "triangulation_covariance_trace": covariance_trace.float(),
        "triangulation_covariance_matrix": covariance_matrix.float(),
        "triangulation_image_only_xyz": image_only_xyz.float(),
        "triangulation_image_only_covariance_trace": (
            image_only_covariance_trace.float()
        ),
        "triangulation_image_only_covariance_matrix": (
            image_only_covariance_matrix.float()
        ),
        "triangulation_image_only_reprojection_median_px": (
            image_only_reprojection_median_px.float()
        ),
        "triangulation_image_only_reprojection_p90_px": (
            image_only_reprojection_p90_px.float()
        ),
        "triangulation_rendered_depth_signed_median_m": (
            rendered_depth_signed_median_m.float()
        ),
        "triangulation_rendered_depth_absolute_median_m": (
            rendered_depth_absolute_median_m.float()
        ),
        "triangulation_rendered_depth_observation_count": (
            rendered_depth_observation_count
        ),
        "triangulation_surface_supported": surface_supported,
        "triangulation_surface_correction_m": surface_correction_m.float(),
        "triangulation_surface_weak_information_ratio": (
            surface_weak_information_ratio.float()
        ),
        "triangulation_surface_depth_improvement_m": (
            surface_depth_improvement_m.float()
        ),
        "triangulation_surface_reprojection_delta_px": (
            surface_reprojection_delta_px.float()
        ),
    }
    torch.set_num_threads(previous_thread_count)
    return result


@torch.no_grad()
def transfer_triangulated_tracks_to_landmarks(
    track_geometry: dict[str, torch.Tensor],
    track_landmark: torch.Tensor,
    landmark_count: int,
    *,
    track_assignment_cost: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Transfer one best independently triangulated track to each landmark."""
    track_xyz = torch.as_tensor(
        track_geometry["triangulated_xyz"], dtype=torch.float32
    ).cpu()
    track_count = int(track_xyz.shape[0])
    track_landmark = torch.as_tensor(track_landmark, dtype=torch.long).cpu()
    if track_landmark.numel() != track_count:
        raise ValueError("track_landmark must have one entry per track")
    if track_assignment_cost is None:
        track_assignment_cost = torch.zeros(track_count, dtype=torch.float32)
    track_assignment_cost = torch.as_tensor(
        track_assignment_cost, dtype=torch.float32
    ).cpu()
    assigned_tracks = torch.nonzero(
        track_landmark >= 0, as_tuple=False
    ).reshape(-1)
    best_track = torch.full((landmark_count,), -1, dtype=torch.long)
    if assigned_tracks.numel() > 0:
        score = track_assignment_cost[assigned_tracks]
        order = torch.argsort(score, stable=True)
        ordered_tracks = assigned_tracks[order]
        ordered_landmarks = track_landmark[ordered_tracks]
        # The distance sort makes the first occurrence the best track for a bank
        # landmark. Stable landmark sorting then exposes those first occurrences.
        landmark_order = torch.argsort(ordered_landmarks, stable=True)
        ordered_tracks = ordered_tracks[landmark_order]
        ordered_landmarks = ordered_landmarks[landmark_order]
        keep = torch.ones(ordered_landmarks.numel(), dtype=torch.bool)
        if keep.numel() > 1:
            keep[1:] = ordered_landmarks[1:] != ordered_landmarks[:-1]
        best_track[ordered_landmarks[keep]] = ordered_tracks[keep]

    bank_geometry = {}
    selected_landmarks = torch.nonzero(
        best_track >= 0, as_tuple=False
    ).reshape(-1)
    selected_tracks = best_track[selected_landmarks]
    for name, value in track_geometry.items():
        value = torch.as_tensor(value).cpu()
        if value.ndim == 0 or value.shape[0] != track_count:
            continue
        shape = (landmark_count, *value.shape[1:])
        if value.dtype == torch.bool:
            default = torch.zeros(shape, dtype=torch.bool)
        elif value.dtype.is_floating_point:
            fill = (
                float("nan")
                if name == "triangulated_xyz"
                else (
                    0.0
                    if name
                    in {
                        "triangulation_parallax_deg",
                    }
                    else float("inf")
                )
            )
            default = torch.full(shape, fill, dtype=value.dtype)
        else:
            default = torch.zeros(shape, dtype=value.dtype)
        default[selected_landmarks] = value[selected_tracks]
        bank_geometry[name] = default
    bank_geometry["track_assignment_cost"] = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float32
    )
    bank_geometry["track_assignment_cost"][selected_landmarks] = (
        track_assignment_cost[selected_tracks]
    )
    bank_geometry["track_assigned"] = best_track >= 0
    assignment = {
        "track_landmark_index": track_landmark,
        "track_assignment_cost": track_assignment_cost,
        "landmark_best_track_index": best_track,
    }
    return bank_geometry, assignment


@torch.no_grad()
def transfer_triangulated_track_groups_to_landmarks(
    track_geometry: dict[str, torch.Tensor],
    *,
    edge_track_index: torch.Tensor,
    edge_landmark_index: torch.Tensor,
    landmark_count: int,
    edge_assignment_cost: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Transfer track evidence through sparse track-to-Gaussian groups."""
    edge_track_index = torch.as_tensor(
        edge_track_index, dtype=torch.long
    ).reshape(-1)
    edge_landmark_index = torch.as_tensor(
        edge_landmark_index, dtype=torch.long
    ).reshape(-1)
    edge_assignment_cost = torch.as_tensor(
        edge_assignment_cost, dtype=torch.float32
    ).reshape(-1)
    if not (
        edge_track_index.numel()
        == edge_landmark_index.numel()
        == edge_assignment_cost.numel()
    ):
        raise ValueError("Track-group edge tensors must have equal lengths")
    track_count = int(
        torch.as_tensor(track_geometry["triangulated_xyz"]).shape[0]
    )
    if edge_track_index.numel() and (
        int(edge_track_index.min()) < 0
        or int(edge_track_index.max()) >= track_count
        or int(edge_landmark_index.min()) < 0
        or int(edge_landmark_index.max()) >= int(landmark_count)
    ):
        raise ValueError("Track-group edge index is out of bounds")

    expanded_geometry = {}
    for name, value in track_geometry.items():
        value = torch.as_tensor(value).cpu()
        if value.ndim > 0 and value.shape[0] == track_count:
            expanded_geometry[name] = value[edge_track_index]
    geometry, edge_assignment = transfer_triangulated_tracks_to_landmarks(
        expanded_geometry,
        edge_landmark_index,
        int(landmark_count),
        track_assignment_cost=edge_assignment_cost,
    )
    landmark_best_edge = edge_assignment["landmark_best_track_index"]
    landmark_best_track = torch.full_like(landmark_best_edge, -1)
    selected = landmark_best_edge >= 0
    landmark_best_track[selected] = edge_track_index[
        landmark_best_edge[selected]
    ]
    edge_order = torch.argsort(edge_landmark_index, stable=True)
    landmark_track_indices = edge_track_index[edge_order]
    landmark_edge_indices = edge_order
    ordered_landmarks = edge_landmark_index[edge_order]
    landmark_track_count = torch.bincount(
        ordered_landmarks, minlength=int(landmark_count)
    )
    landmark_track_offsets = torch.zeros(
        int(landmark_count) + 1, dtype=torch.long
    )
    landmark_track_offsets[1:] = torch.cumsum(
        landmark_track_count, dim=0
    )
    raw_responsibility = (
        1.0 - edge_assignment_cost[edge_order]
    ).clamp_min(1e-6)
    responsibility_sum = torch.zeros(
        int(landmark_count), dtype=torch.float32
    )
    responsibility_sum.index_add_(
        0, ordered_landmarks, raw_responsibility
    )
    landmark_track_responsibilities = raw_responsibility / (
        responsibility_sum[ordered_landmarks].clamp_min(1e-12)
    )
    responsibility_square_sum = torch.zeros_like(responsibility_sum)
    responsibility_square_sum.index_add_(
        0,
        ordered_landmarks,
        landmark_track_responsibilities.square(),
    )
    effective_track_support = torch.zeros_like(responsibility_sum)
    has_support = landmark_track_count > 0
    effective_track_support[has_support] = (
        1.0 / responsibility_square_sum[has_support].clamp_min(1e-12)
    )
    track_xyz = torch.as_tensor(
        track_geometry["triangulated_xyz"], dtype=torch.float32
    ).cpu()
    landmark_track_xyz_mean = torch.zeros(
        (int(landmark_count), 3), dtype=torch.float32
    )
    landmark_track_xyz_mean.index_add_(
        0,
        ordered_landmarks,
        track_xyz[landmark_track_indices]
        * landmark_track_responsibilities[:, None],
    )
    edge_residual = torch.linalg.norm(
        track_xyz[landmark_track_indices]
        - landmark_track_xyz_mean[ordered_landmarks],
        dim=1,
    )
    weighted_square_residual = torch.zeros_like(responsibility_sum)
    weighted_square_residual.index_add_(
        0,
        ordered_landmarks,
        landmark_track_responsibilities * edge_residual.square(),
    )
    landmark_track_xyz_rms_m = torch.sqrt(weighted_square_residual)
    landmark_track_xyz_max_residual_m = torch.zeros_like(
        responsibility_sum
    )
    landmark_track_xyz_max_residual_m.scatter_reduce_(
        0,
        ordered_landmarks,
        edge_residual,
        reduce="amax",
        include_self=True,
    )
    geometry.update(
        {
            "landmark_track_count": landmark_track_count,
            "landmark_effective_track_support": effective_track_support,
            "landmark_track_xyz_mean": landmark_track_xyz_mean,
            "landmark_track_xyz_rms_m": landmark_track_xyz_rms_m,
            "landmark_track_xyz_max_residual_m": (
                landmark_track_xyz_max_residual_m
            ),
        }
    )
    assignment = {
        "edge_track_index": edge_track_index,
        "edge_landmark_index": edge_landmark_index,
        "edge_assignment_cost": edge_assignment_cost,
        "landmark_best_edge_index": landmark_best_edge,
        "landmark_best_track_index": landmark_best_track,
        "landmark_track_offsets": landmark_track_offsets,
        "landmark_track_indices": landmark_track_indices,
        "landmark_track_edge_indices": landmark_edge_indices,
        "landmark_track_responsibilities": (
            landmark_track_responsibilities
        ),
    }
    return geometry, assignment


@torch.no_grad()
def assign_triangulated_tracks_to_landmarks(
    track_geometry: dict[str, torch.Tensor],
    bank_xyz: torch.Tensor,
    *,
    maximum_distance_m: float = 0.20,
    minimum_margin_m: float = 0.0,
    require_high_confidence: bool = True,
    device: str | torch.device = "cuda",
    chunk_size: int = 512,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Associate independent 2D tracks to frozen bank geometry after triangulation."""
    bank_xyz_cpu = torch.as_tensor(bank_xyz, dtype=torch.float32).cpu()
    track_xyz = torch.as_tensor(
        track_geometry["triangulated_xyz"], dtype=torch.float32
    ).cpu()
    eligible = torch.as_tensor(
        track_geometry[
            "triangulation_high_confidence"
            if require_high_confidence
            else "triangulated"
        ],
        dtype=torch.bool,
    ).reshape(-1)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    track_landmark = torch.full(
        (track_xyz.shape[0],), -1, dtype=torch.long
    )
    track_landmark_distance = torch.full(
        (track_xyz.shape[0],), float("inf"), dtype=torch.float32
    )
    track_landmark_margin = torch.zeros(
        track_xyz.shape[0], dtype=torch.float32
    )
    if eligible_indices.numel() > 0:
        bank_device = bank_xyz_cpu.to(device)
        for start in range(0, eligible_indices.numel(), max(int(chunk_size), 1)):
            selected = eligible_indices[start : start + max(int(chunk_size), 1)]
            distance = torch.cdist(track_xyz[selected].to(device), bank_device)
            nearest_two, nearest_indices = torch.topk(
                distance, k=min(2, distance.shape[1]), dim=1, largest=False
            )
            nearest_distance = nearest_two[:, 0]
            nearest = nearest_indices[:, 0]
            if nearest_two.shape[1] > 1:
                margin = nearest_two[:, 1] - nearest_two[:, 0]
            else:
                margin = torch.full_like(nearest_distance, float("inf"))
            valid = (
                (nearest_distance <= float(maximum_distance_m))
                & (margin >= float(minimum_margin_m))
            )
            track_landmark[selected[valid.cpu()]] = nearest[valid].cpu()
            track_landmark_distance[selected[valid.cpu()]] = (
                nearest_distance[valid].cpu()
            )
            track_landmark_margin[selected] = margin.cpu()
    geometry, assignment = transfer_triangulated_tracks_to_landmarks(
        track_geometry,
        track_landmark,
        int(bank_xyz_cpu.shape[0]),
        track_assignment_cost=track_landmark_distance,
    )
    geometry["track_assignment_distance_m"] = geometry.pop(
        "track_assignment_cost"
    )
    best_track = assignment["landmark_best_track_index"]
    selected_landmark = best_track >= 0
    geometry["track_assignment_margin_m"] = torch.zeros(
        bank_xyz_cpu.shape[0], dtype=torch.float32
    )
    geometry["track_assignment_margin_m"][selected_landmark] = (
        track_landmark_margin[best_track[selected_landmark]]
    )
    assignment["track_landmark_distance_m"] = assignment.pop(
        "track_assignment_cost"
    )
    assignment["track_landmark_margin_m"] = track_landmark_margin
    return geometry, assignment
