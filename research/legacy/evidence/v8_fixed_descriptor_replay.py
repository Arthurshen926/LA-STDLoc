"""Exact F0 fusion replay from original-cache CSR and association confidence."""
import torch
from evidence.tracks import fuse_projective_anchor_observations


def prepare_replay(state, cache, association):
    names = list(cache['queries'])
    if names != state['v6_mapping_query_names'] or names != association['query_names']:
        raise ValueError('query registries differ')
    if not torch.equal(state['v6_mapping_query_bins'], association['query_bins']):
        raise ValueError('fusion bins differ')
    if state['provenance'].get('joint_track_purification', {}).get('enabled', False):
        raise ValueError('replay only supports unpurified F0')
    records = [cache['queries'][n] for n in names]
    counts = torch.tensor([len(r['native_keypoints']) for r in records])
    raw_offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    source = association['source_keypoint_indices_by_query']
    source_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor([len(r) for r in source]).cumsum(0)])
    tracks = association['tracks']
    tq, tk = tracks['query_index'].long(), tracks['keypoint_index'].long()
    original_row = torch.cat(source)[source_offsets[tq]+tk]
    packed = raw_offsets[tq]+original_row
    if torch.unique(packed).numel() != packed.numel():
        raise ValueError('multiple associations share a source observation')
    confidence = torch.full((int(raw_offsets[-1]),), float('nan'))
    confidence[packed] = tracks['confidence'].float()
    csr = state['projective_anchor_observations']
    if csr.get('keypoint_index_semantics') != 'original_unfiltered_observation_cache_row':
        raise ValueError('source row semantics required')
    q, k = csr['query_indices'].long(), csr['keypoint_indices'].long()
    offsets = csr['observation_offsets'].long()
    if offsets.shape != (len(state['anchor_ids'])+1,) or offsets[0] != 0 or offsets[-1] != len(q) or torch.any(offsets[1:]<=offsets[:-1]):
        raise ValueError('invalid CSR offsets')
    if q.shape != k.shape or torch.any(q<0) or torch.any(q>=len(names)) or torch.any(k<0) or torch.any(k>=counts[q]):
        raise ValueError('invalid source row IDs')
    source_index = raw_offsets[q]+k
    detector = torch.cat([r['native_scores'].float() for r in records])[source_index]
    weights = confidence[source_index]
    for a, kind in enumerate(state['anchor_candidate_kind']):
        start, stop = int(offsets[a]), int(offsets[a+1])
        if kind == 'depth_proposed_projective_completion':
            if torch.isfinite(weights[start:stop]).any():
                raise ValueError('completion reuses base Track observations')
            weights[start:stop] = detector[start:stop].clamp(0,1)
        elif kind == 'v2_projective_track':
            if not torch.isfinite(weights[start:stop]).all():
                raise ValueError('base confidence missing')
        else:
            raise ValueError('unsupported candidate kind')
    return {'source_index': source_index, 'weights': detector*weights,
            'bins': association['query_bins'][q], 'offsets': offsets}


def fuse_replay(descriptors, replay):
    observations = torch.cat(descriptors)[replay['source_index']]
    result = []
    for start, stop in zip(replay['offsets'][:-1], replay['offsets'][1:]):
        result.append(fuse_projective_anchor_observations(
            observations[start:stop], replay['bins'][start:stop],
            detector_weight=replay['weights'][start:stop], trim_fraction=.2))
    return torch.stack(result)


def fuse_geometry_balanced(descriptors, bins, weights, reliability):
    """Retain camera-bin balance while carrying geometric quality between bins.

    Legacy fusion normalizes each prototype, cancelling absolute reliability
    for singleton bins. Here the weighted mean reliability survives separately.
    This is an experimental fusion rule, not a certified physical error weight.
    """
    from torch.nn import functional as F
    descriptors = F.normalize(descriptors.float(), dim=1)
    if weights.shape != bins.shape or reliability.shape != bins.shape or len(descriptors) != len(bins):
        raise ValueError('fusion evidence shapes differ')
    if not torch.isfinite(reliability).all() or torch.any(reliability<0) or torch.any(reliability>1):
        raise ValueError('reliability must be finite in [0,1]')
    prototypes, qualities = [], []
    for b in torch.unique(bins, sorted=True):
        select = bins == b
        w = weights[select].clamp_min(1e-4)
        r = reliability[select]
        prototypes.append(F.normalize((descriptors[select]*(w*r).clamp_min(1e-4)[:,None]).sum(0),dim=0))
        qualities.append((w*r).sum()/w.sum())
    prototypes, quality = torch.stack(prototypes), torch.stack(qualities)
    similarity = prototypes @ prototypes.T
    medoid = int(similarity.mean(1).argmax())
    keep = torch.topk(similarity[medoid], max(1,round(len(prototypes)*.8))).indices
    if quality[keep].sum()<=1e-8:
        return fuse_projective_anchor_observations(descriptors,bins,detector_weight=weights,trim_fraction=.2)
    return F.normalize((prototypes[keep]*quality[keep,None]).sum(0),dim=0)
