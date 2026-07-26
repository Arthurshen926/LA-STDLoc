from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def _camera_centers_and_axes(
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float64)
    camera_centers = -torch.einsum(
        "qji,qj->qi", pose_w2c[:, :3, :3], pose_w2c[:, :3, 3]
    )
    optical_axis = torch.einsum(
        "qji,j->qi",
        pose_w2c[:, :3, :3],
        pose_w2c.new_tensor([0.0, 0.0, 1.0]),
    )
    return camera_centers, F.normalize(optical_axis, dim=1)


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


def candidate_camera_pairs(
    pose_w2c: torch.Tensor,
    *,
    neighbors: int = 6,
    minimum_baseline_m: float = 0.03,
    maximum_baseline_m: float = 5.0,
    maximum_axis_angle_deg: float = 75.0,
) -> list[tuple[int, int]]:
    """Build a deterministic local view graph without descriptor/map IDs."""
    centers, axes = _camera_centers_and_axes(pose_w2c)
    count = int(centers.shape[0])
    if count < 2:
        return []
    distance = torch.cdist(centers, centers)
    axis_cosine = (axes @ axes.T).clamp(-1.0, 1.0)
    minimum_cosine = float(
        torch.cos(torch.deg2rad(torch.tensor(maximum_axis_angle_deg))).item()
    )
    valid = (
        (distance >= float(minimum_baseline_m))
        & (distance <= float(maximum_baseline_m))
        & (axis_cosine >= minimum_cosine)
    )
    valid.fill_diagonal_(False)
    positive = distance[valid]
    distance_scale = positive.median().clamp_min(1e-6) if positive.numel() else 1.0
    cost = distance / distance_scale + 0.5 * (1.0 - axis_cosine)
    cost = cost.masked_fill(~valid, torch.inf)
    pairs = set()
    width = min(max(int(neighbors), 1), max(count - 1, 1))
    for query in range(count):
        candidates = torch.topk(
            cost[query], width, largest=False, sorted=True
        ).indices
        for other in candidates.tolist():
            if not bool(torch.isfinite(cost[query, other])):
                continue
            pairs.add((min(query, other), max(query, other)))
    return sorted(pairs)


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
    uv_b = torch.as_tensor(uv_b, dtype=torch.float64)
    fundamental = torch.as_tensor(fundamental, dtype=torch.float64)
    ones = torch.ones((uv_a.shape[0], 1), dtype=uv_a.dtype)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match native descriptors using reciprocal, margin and epipolar gates."""
    if descriptors_a.numel() == 0 or descriptors_b.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, torch.empty(0)
    descriptors_a = F.normalize(descriptors_a.float(), dim=1)
    descriptors_b = F.normalize(descriptors_b.float(), dim=1)
    similarity = descriptors_a @ descriptors_b.T
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
        return empty, empty, torch.empty(0)
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
    return selected.long(), selected_target.long(), confidence


@torch.no_grad()
def local_geometric_match_support(
    uv_a: torch.Tensor,
    uv_b: torch.Tensor,
    *,
    neighbors: int = 8,
    angle_threshold_cosine: float = 0.9659,
    scale_threshold: float = 0.1,
    scale_limit: float = 3.0,
    maximum_edge_px: float = 50.0,
) -> torch.Tensor:
    """Score map-independent local triangle consistency for 2D-2D matches."""
    uv_a = torch.as_tensor(uv_a, dtype=torch.float32)
    uv_b = torch.as_tensor(uv_b, device=uv_a.device, dtype=uv_a.dtype)
    if uv_a.ndim != 2 or uv_b.shape != uv_a.shape or uv_a.shape[1] != 2:
        raise ValueError("Local geometry expects aligned [N, 2] coordinates")
    count = int(uv_a.shape[0])
    if count < 3:
        return uv_a.new_zeros(count)
    neighbors = min(max(int(neighbors), 1), count - 1)
    if neighbors < 2:
        return uv_a.new_zeros(count)
    if not 0.0 < float(angle_threshold_cosine) <= 1.0:
        raise ValueError("angle_threshold_cosine must be in (0, 1]")
    if float(scale_threshold) <= 0.0:
        raise ValueError("scale_threshold must be positive")
    if float(scale_limit) <= 1.0:
        raise ValueError("scale_limit must exceed one")

    distance = torch.cdist(uv_a, uv_a)
    nearest = distance.topk(
        neighbors + 1, dim=-1, largest=False
    ).indices[:, 1:]
    relative_a = uv_a[nearest] - uv_a[:, None, :]
    relative_b = uv_b[nearest] - uv_b[:, None, :]

    unit_a = F.normalize(relative_a, dim=-1, eps=1e-8)
    unit_b = F.normalize(relative_b, dim=-1, eps=1e-8)
    angle_a = torch.matmul(unit_a, unit_a.transpose(1, 2))
    angle_b = torch.matmul(unit_b, unit_b.transpose(1, 2))
    angle_consistent = (
        angle_a - angle_b
    ).abs() < 1.0 - float(angle_threshold_cosine)

    edge_a_j = relative_a[:, :, None, :]
    edge_a_k = relative_a[:, None, :, :]
    edge_b_j = relative_b[:, :, None, :]
    edge_b_k = relative_b[:, None, :, :]
    cross_a = (
        edge_a_j[..., 0] * edge_a_k[..., 1]
        - edge_a_j[..., 1] * edge_a_k[..., 0]
    )
    cross_b = (
        edge_b_j[..., 0] * edge_b_k[..., 1]
        - edge_b_j[..., 1] * edge_b_k[..., 0]
    )
    orientation_consistent = cross_a * cross_b > 0.0

    length_a = relative_a.norm(dim=-1)
    length_b = relative_b.norm(dim=-1)
    invalid_edge = (
        (length_a < 1e-6)
        | (length_b < 1e-6)
        | (length_a > float(maximum_edge_px))
        | (length_b > float(maximum_edge_px))
    )
    invalid_triangle = invalid_edge[:, :, None] | invalid_edge[:, None, :]
    opposite_a = (edge_a_j - edge_a_k).norm(dim=-1)
    opposite_b = (edge_b_j - edge_b_k).norm(dim=-1)
    invalid_triangle |= (opposite_a < 1e-6) | (opposite_b < 1e-6)

    scale_a = edge_b_j.norm(dim=-1) / edge_a_j.norm(dim=-1).clamp_min(1e-8)
    scale_b = edge_b_k.norm(dim=-1) / edge_a_k.norm(dim=-1).clamp_min(1e-8)
    scale_c = opposite_b / opposite_a.clamp_min(1e-8)
    scale_consistent = (
        ((scale_a - scale_b).abs() < float(scale_threshold))
        & ((scale_a - scale_c).abs() < float(scale_threshold))
        & ((scale_b - scale_c).abs() < float(scale_threshold))
    )
    lower = 1.0 / float(scale_limit)
    upper = float(scale_limit)
    scale_consistent &= (
        (scale_a > lower)
        & (scale_a < upper)
        & (scale_b > lower)
        & (scale_b < upper)
        & (scale_c > lower)
        & (scale_c < upper)
    )

    support = (
        angle_consistent
        & orientation_consistent
        & scale_consistent
        & ~invalid_triangle
    )
    diagonal = torch.arange(neighbors, device=uv_a.device)
    support[:, diagonal, diagonal] = False
    return support.float().sum(dim=(1, 2))


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


@torch.no_grad()
def build_cycle_consistent_tracks(
    *,
    descriptors: list[torch.Tensor],
    keypoints: list[torch.Tensor],
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    detector_scores: list[torch.Tensor] | None = None,
    pair_neighbors: int = 6,
    minimum_baseline_m: float = 0.03,
    maximum_baseline_m: float = 5.0,
    maximum_axis_angle_deg: float = 75.0,
    minimum_similarity: float = 0.65,
    minimum_margin: float = 0.01,
    maximum_epipolar_error_px: float = 2.0,
    local_geometry_filter: bool = False,
    local_geometry_neighbors: int = 8,
    local_geometry_support_threshold: float = 4.0,
    local_geometry_angle_cosine: float = 0.9659,
    local_geometry_scale_threshold: float = 0.1,
    local_geometry_scale_limit: float = 3.0,
    local_geometry_maximum_edge_px: float = 50.0,
    local_geometry_minimum_matches: int = 8,
    local_geometry_mode: str = "hard",
    local_geometry_confidence_floor: float = 0.25,
    minimum_track_views: int = 3,
    require_cycle: bool = True,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    """Build map-independent native 2D tracks from a local camera graph."""
    query_count = len(descriptors)
    if len(keypoints) != query_count:
        raise ValueError("Descriptor and keypoint camera tables must align")
    camera_K = torch.as_tensor(camera_K, dtype=torch.float64).cpu()
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float64).cpu()
    if local_geometry_mode not in {"hard", "soft"}:
        raise ValueError("local_geometry_mode must be 'hard' or 'soft'")
    if not 0.0 <= float(local_geometry_confidence_floor) <= 1.0:
        raise ValueError("local_geometry_confidence_floor must be in [0, 1]")
    pairs = candidate_camera_pairs(
        pose_w2c,
        neighbors=pair_neighbors,
        minimum_baseline_m=minimum_baseline_m,
        maximum_baseline_m=maximum_baseline_m,
        maximum_axis_angle_deg=maximum_axis_angle_deg,
    )
    pair_matches = {}
    raw_match_count = 0
    local_geometry_eligible_pairs = 0
    local_geometry_skipped_pairs = 0
    local_geometry_accepted_edges = 0
    local_geometry_rejected_edges = 0
    local_geometry_low_support_edges = 0
    local_geometry_support_sum = 0.0
    local_geometry_support_count = 0
    device = torch.device(device)
    for left, right in pairs:
        source, target, confidence = reciprocal_epipolar_matches(
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
        )
        if source.numel() == 0:
            continue
        raw_match_count += int(source.numel())
        if bool(local_geometry_filter):
            if source.numel() >= int(local_geometry_minimum_matches):
                support = local_geometric_match_support(
                    torch.as_tensor(keypoints[left])[source],
                    torch.as_tensor(keypoints[right])[target],
                    neighbors=local_geometry_neighbors,
                    angle_threshold_cosine=local_geometry_angle_cosine,
                    scale_threshold=local_geometry_scale_threshold,
                    scale_limit=local_geometry_scale_limit,
                    maximum_edge_px=local_geometry_maximum_edge_px,
                )
                keep = support >= float(local_geometry_support_threshold)
                local_geometry_eligible_pairs += 1
                local_geometry_low_support_edges += int((~keep).sum().item())
                local_geometry_support_sum += float(support.sum().item())
                local_geometry_support_count += int(support.numel())
                if local_geometry_mode == "hard":
                    local_geometry_accepted_edges += int(keep.sum().item())
                    local_geometry_rejected_edges += int((~keep).sum().item())
                    source = source[keep]
                    target = target[keep]
                    confidence = confidence[keep]
                else:
                    normalized_support = (
                        support / max(float(local_geometry_support_threshold), 1.0)
                    ).clamp(0.0, 1.0)
                    confidence_weight = float(
                        local_geometry_confidence_floor
                    ) + (
                        1.0 - float(local_geometry_confidence_floor)
                    ) * normalized_support
                    confidence = confidence * confidence_weight
                    local_geometry_accepted_edges += int(source.numel())
            else:
                local_geometry_skipped_pairs += 1
                local_geometry_accepted_edges += int(source.numel())
        if source.numel() == 0:
            continue
        if detector_scores is not None:
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
    node_query = {}
    for query, (start, end) in enumerate(
        zip(keypoint_offsets[:-1], keypoint_offsets[1:])
    ):
        for node in range(start, end):
            if node in disjoint.parent:
                node_query[node] = (query, node - start)
    track_indices = []
    query_indices = []
    keypoint_indices = []
    confidences = []
    track_count = 0
    rejected_duplicate_query = 0
    for nodes in components.values():
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
        track_count += 1
    tracks = {
        "track_index": torch.as_tensor(track_indices, dtype=torch.long),
        "query_index": torch.as_tensor(query_indices, dtype=torch.long),
        "keypoint_index": torch.as_tensor(keypoint_indices, dtype=torch.long),
        "confidence": torch.as_tensor(confidences, dtype=torch.float32),
    }
    diagnostics = {
        "track_camera_pair_candidate_count": len(pairs),
        "track_camera_pair_matched_count": len(pair_matches),
        "track_raw_reciprocal_epipolar_edge_count": raw_match_count,
        "track_lgcv_enabled": int(bool(local_geometry_filter)),
        "track_lgcv_mode": (
            str(local_geometry_mode) if local_geometry_filter else "disabled"
        ),
        "track_lgcv_confidence_floor": float(
            local_geometry_confidence_floor
        ),
        "track_lgcv_eligible_pair_count": local_geometry_eligible_pairs,
        "track_lgcv_skipped_pair_count": local_geometry_skipped_pairs,
        "track_lgcv_accepted_edge_count": local_geometry_accepted_edges,
        "track_lgcv_rejected_edge_count": local_geometry_rejected_edges,
        "track_lgcv_low_support_edge_count": (
            local_geometry_low_support_edges
        ),
        "track_lgcv_support_mean": (
            local_geometry_support_sum / max(local_geometry_support_count, 1)
        ),
        "track_cycle_supported_edge_count": supported_edge_count,
        "track_count": track_count,
        "track_observation_count": len(track_indices),
        "track_rejected_duplicate_query_component_count": (
            rejected_duplicate_query
        ),
    }
    return tracks, diagnostics


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
        triangulated_xyz[landmark] = point
        observation_count[landmark] = int(finite.sum())
        reprojection_median_px[landmark] = residual.median()
        reprojection_p90_px[landmark] = torch.quantile(residual, 0.9)
        condition_number[landmark] = condition
        covariance_trace[landmark] = torch.trace(covariance)
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
        "triangulation_rendered_depth_signed_median_m": (
            rendered_depth_signed_median_m.float()
        ),
        "triangulation_rendered_depth_absolute_median_m": (
            rendered_depth_absolute_median_m.float()
        ),
        "triangulation_rendered_depth_observation_count": (
            rendered_depth_observation_count
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
    assignment = {
        "edge_track_index": edge_track_index,
        "edge_landmark_index": edge_landmark_index,
        "edge_assignment_cost": edge_assignment_cost,
        "landmark_best_edge_index": landmark_best_edge,
        "landmark_best_track_index": landmark_best_track,
    }
    return geometry, assignment


@torch.no_grad()
def assign_triangulated_tracks_to_landmarks(
    track_geometry: dict[str, torch.Tensor],
    bank_xyz: torch.Tensor,
    *,
    maximum_distance_m: float = 0.20,
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
    if eligible_indices.numel() > 0:
        bank_device = bank_xyz_cpu.to(device)
        for start in range(0, eligible_indices.numel(), max(int(chunk_size), 1)):
            selected = eligible_indices[start : start + max(int(chunk_size), 1)]
            distance = torch.cdist(track_xyz[selected].to(device), bank_device)
            nearest_distance, nearest = distance.min(dim=1)
            valid = nearest_distance <= float(maximum_distance_m)
            track_landmark[selected[valid.cpu()]] = nearest[valid].cpu()
            track_landmark_distance[selected[valid.cpu()]] = (
                nearest_distance[valid].cpu()
            )
    geometry, assignment = transfer_triangulated_tracks_to_landmarks(
        track_geometry,
        track_landmark,
        int(bank_xyz_cpu.shape[0]),
        track_assignment_cost=track_landmark_distance,
    )
    geometry["track_assignment_distance_m"] = geometry.pop(
        "track_assignment_cost"
    )
    assignment["track_landmark_distance_m"] = assignment.pop(
        "track_assignment_cost"
    )
    return geometry, assignment
