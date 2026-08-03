"""Deterministic mapping-camera support/query splits."""

from __future__ import annotations

import torch


def _camera_sequence_key(camera) -> str:
    image_name = str(getattr(camera, "image_name", "")).replace("\\", "/")
    return image_name.split("/", 1)[0] if "/" in image_name else ""


def _query_count(camera_count: int, query_ratio: float) -> int:
    ratio = max(0.0, min(1.0, float(query_ratio)))
    return max(1, min(camera_count - 1, int(round(camera_count * ratio))))


def _split(cameras, query_ids):
    support = [camera for index, camera in enumerate(cameras) if index not in query_ids]
    query = [camera for index, camera in enumerate(cameras) if index in query_ids]
    return support, query


def _sequence_block(cameras, count, generator):
    groups = {}
    for index, camera in enumerate(cameras):
        groups.setdefault(_camera_sequence_key(camera), []).append(index)
    if len(groups) < 2:
        return None
    keys = list(groups)
    order = torch.randperm(len(keys), generator=generator).tolist()
    selected = set()
    for position in order:
        candidate = groups[keys[position]]
        if len(selected) + len(candidate) >= len(cameras):
            continue
        selected.update(candidate)
        if len(selected) >= count:
            break
    return selected or None


def _temporal_block(camera_count, count, generator):
    maximum_start = max(0, camera_count - count)
    start = int(torch.randint(maximum_start + 1, (1,), generator=generator))
    return set(range(start, start + count))


def _stratified_temporal_block(cameras, count, generator):
    sequence_to_indices = {}
    for index, camera in enumerate(cameras):
        sequence_to_indices.setdefault(_camera_sequence_key(camera), []).append(index)
    groups = [indices for indices in sequence_to_indices.values() if len(indices) >= 2]
    if len(groups) < 2:
        return _temporal_block(len(cameras), count, generator)
    capacity = [len(indices) - 1 for indices in groups]
    target = min(int(count), sum(capacity))
    if target <= 0:
        return set()
    total = sum(len(indices) for indices in groups)
    ideal = [target * len(indices) / total for indices in groups]
    allocation = [min(cap, int(value)) for cap, value in zip(capacity, ideal)]
    remaining = target - sum(allocation)
    tie_order = torch.randperm(len(groups), generator=generator).tolist()
    tie_rank = {group: rank for rank, group in enumerate(tie_order)}
    priority = sorted(
        range(len(groups)),
        key=lambda group: (ideal[group] - allocation[group], -tie_rank[group]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for group in priority:
            if allocation[group] >= capacity[group]:
                continue
            allocation[group] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    selected = set()
    for indices, group_count in zip(groups, allocation):
        if group_count <= 0:
            continue
        start = int(
            torch.randint(
                len(indices) - group_count + 1,
                (1,),
                generator=generator,
            )
        )
        selected.update(indices[start : start + group_count])
    return selected


def split_support_query_cameras(
    cameras,
    query_ratio: float = 0.2,
    seed: int = 0,
    mode: str = "random",
):
    """Split mapping cameras without ever introducing test cameras."""
    cameras = list(cameras)
    if len(cameras) < 2:
        return cameras, cameras.copy()
    count = _query_count(len(cameras), query_ratio)
    generator = torch.Generator().manual_seed(int(seed))
    if mode == "random":
        query_ids = set(
            torch.randperm(len(cameras), generator=generator)[:count].tolist()
        )
    elif mode == "sequence_block":
        query_ids = _sequence_block(cameras, count, generator)
        if query_ids is None:
            query_ids = set(
                torch.randperm(len(cameras), generator=generator)[:count].tolist()
            )
    elif mode == "temporal_block":
        query_ids = _temporal_block(len(cameras), count, generator)
    elif mode == "stratified_temporal_block":
        query_ids = _stratified_temporal_block(cameras, count, generator)
    else:
        raise ValueError(f"unknown support/query split mode: {mode}")
    return _split(cameras, query_ids)
