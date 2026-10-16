"""Fixed, deterministic selection and descriptor controls used in the paper."""

import numpy as np
import torch
import torch.nn.functional as F


def spatial_rows(xyz, k, voxel_m=0.5):
    """Voxel round-robin, exact K, independent of scores and query labels.

    Within a voxel use distance to its geometric center, then source row.
    At a partially filled round use evenly spaced occupied-voxel ranks.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if not 0 < k <= len(xyz) or not np.isfinite(xyz).all() or voxel_m <= 0:
        raise ValueError("invalid spatial selection input")
    cells = np.floor(xyz / voxel_m).astype(np.int64)
    _, group = np.unique(cells, axis=0, return_inverse=True)
    d2 = ((xyz - (cells + 0.5) * voxel_m) ** 2).sum(1)
    order = np.lexsort((np.arange(len(xyz)), d2, group))
    counts = np.bincount(group)
    starts = np.r_[0, counts.cumsum()[:-1]]
    chosen = []
    left = k
    for level in range(int(counts.max())):
        active = np.flatnonzero(counts > level)
        if len(active) > left:
            active = active[
                np.floor((np.arange(left) + 0.5) * len(active) / left).astype(int)
            ]
        chosen.extend(order[starts[active] + level].tolist())
        left -= len(active)
        if left == 0:
            break
    result = np.sort(np.asarray(chosen, dtype=np.int64))
    assert len(result) == k and len(np.unique(result)) == k
    return result


def quality_rows(quality, k):
    quality = np.asarray(quality)
    if not 0 < k <= len(quality) or not np.isfinite(quality).all():
        raise ValueError("invalid quality selection input")
    return np.sort(np.lexsort((np.arange(len(quality)), -quality))[:k])


def single_descriptors(csr, names, records):
    off = np.asarray(csr["observation_offsets"])
    q = np.asarray(csr["query_indices"])
    k = np.asarray(csr["keypoint_indices"])
    values = []
    chosen = []
    for a, b in zip(off[:-1], off[1:]):
        if a == b:
            raise ValueError("empty track")
        j = min(range(int(a), int(b)), key=lambda j: (names[int(q[j])], int(k[j])))
        values.append(
            torch.as_tensor(records[names[int(q[j])]]["native_descriptors"])[
                int(k[j])
            ].float()
        )
        chosen.append((int(q[j]), int(k[j])))
    return F.normalize(torch.stack(values), dim=1), np.asarray(chosen, dtype=np.int64)
