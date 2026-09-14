"""Mapping-only coverage/geometry audit; no queries and no map mutation."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901')
    reports = {}
    for scene in ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
        m = torch.load(root / scene / 'projective_map/projective_anchor_map.pt',
                       map_location='cpu', weights_only=False)
        assert m['provenance']['uses_test_queries'] is False
        csr = m['projective_anchor_observations']
        support = csr['observation_offsets'][1:] - csr['observation_offsets'][:-1]
        covariance = m['anchor_position_covariance'].float()
        finite = torch.isfinite(covariance).reshape(len(covariance), -1).all(dim=1)
        sigma = torch.linalg.eigvalsh(covariance[finite]).amax(dim=1).clamp_min(0).sqrt()
        quantiles = torch.tensor([0., .25, .5, .75, .9, .99, 1.])
        spatial = {}
        for size in [.02, .05, .1]:
            _, counts = torch.unique(torch.floor(m['anchor_xyz'] / size).long(),
                                     dim=0, return_counts=True)
            spatial[str(size)] = dict(occupied_voxels=len(counts),
                                     multi_anchor_voxels=int((counts > 1).sum()))
        # A conservative redundancy proposal: close XYZ, near-identical descriptor,
        # and all supporting mapping cameras already represented by the survivor.
        # This is only an audit, not a proof of equal novel-view coverage.
        from scipy.spatial import cKDTree
        import numpy as np
        xyz = m['anchor_xyz'].numpy()
        features = torch.nn.functional.normalize(m['anchor_features'].float(), dim=1).numpy()
        offsets = csr['observation_offsets'].tolist()
        queries = csr['query_indices'].tolist()
        cameras = [set(queries[offsets[i]:offsets[i+1]]) for i in range(len(support))]
        tree = cKDTree(xyz)
        duplicate_proposals = {}
        for radius, cosine in [(.02, .98), (.05, .95)]:
            kept = np.ones(len(support), dtype=bool)
            pairs = []
            order = sorted(range(len(support)), key=lambda i: (-len(cameras[i]), i))
            for i in order:
                if not kept[i]:
                    continue
                for j in sorted(tree.query_ball_point(xyz[i], radius)):
                    if j == i or not kept[j] or not cameras[j].issubset(cameras[i]):
                        continue
                    if float(features[i] @ features[j]) >= cosine:
                        kept[j] = False
                        pairs.append([j, i])
            duplicate_proposals[f'r{radius}_cos{cosine}'] = dict(
                removed=len(pairs), retained=int(kept.sum()), removed_to_survivor=pairs)
        reports[scene] = dict(anchor_count=len(support),
                             support_quantiles=torch.quantile(support.float(), quantiles).tolist(),
                             sigma_m_quantiles=torch.quantile(sigma, quantiles).tolist(),
                             nonfinite_covariance=int((~finite).sum()),
                             spatial=spatial,
                             duplicate_proposals=duplicate_proposals,
                             warning='Spatial co-location is not descriptor redundancy; '
                                     'do not prune from these counts alone.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(mapping_only=True, map_mutated=False,
                                       reports=reports), indent=2))


if __name__ == '__main__':
    main()
