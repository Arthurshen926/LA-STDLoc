"""Single observed descriptor selection, with no query or geometry feedback."""
import torch
import torch.nn.functional as F


def representatives(descriptors, bins, weights, reference):
    """Return bin-balanced cosine medoid and nearest-to-F0 observation.

    Equal total mass per camera bin prevents dense capture trajectories from
    dominating the medoid. Computing dot products against the weighted mean
    exactly maximizes weighted pairwise cosine similarity, without an NxN array.
    Ties use the first CSR observation (deterministic, not permutation invariant).
    """
    if descriptors.ndim != 2 or len(descriptors) == 0:
        raise ValueError('nonempty descriptor matrix required')
    if bins.shape != weights.shape or len(bins) != len(descriptors):
        raise ValueError('observation shapes differ')
    if not torch.isfinite(descriptors).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('finite descriptors and nonnegative weights required')
    d = F.normalize(descriptors.float(), dim=1)
    _, inverse = torch.unique(bins, sorted=True, return_inverse=True)
    w = weights.float().clamp_min(1e-4)
    mass = torch.zeros(int(inverse.max()) + 1, dtype=w.dtype, device=w.device)
    mass.scatter_add_(0, inverse, w)
    balanced = w / mass[inverse]
    center = (d * balanced[:, None]).sum(0)
    indices = torch.stack([(d @ center).argmax(), (d @ reference.float()).argmax()])
    return d[indices], indices
