"""Mapping-only observation factorial; selection runs AFTER observation changes."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from anygsloc.artifacts import export_map, sha256, validate_identity
from anygsloc.mapping.compact import select_compact_rows
from anygsloc.mapping.subset import subset


def representative_features(state, records):
    names = state['v6_mapping_query_names']
    csr = state['projective_anchor_observations']
    offsets = csr['observation_offsets'].tolist()
    queries = csr['query_indices'].tolist()
    keypoints = csr['keypoint_indices'].tolist()
    descriptors = [records[n]['native_descriptors'] for n in names]
    result = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        if start == end:
            raise ValueError('Anchor without observations')
        j = min(range(start, end), key=lambda j: (names[queries[j]], keypoints[j]))
        result.append(descriptors[queries[j]][keypoints[j]].float())
    return F.normalize(torch.stack(result), dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map-dir', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fusion', type=int, choices=[0, 1], required=True)
    p.add_argument('--completion', type=int, choices=[0, 1], required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.output.exists():
        raise FileExistsError(a.output)
    source = a.map_dir / 'projective_anchor_map.pt'
    state = torch.load(source, map_location='cpu', weights_only=False)
    metric = torch.load(a.map_dir / 'identity_metric.pt', map_location='cpu', weights_only=False)
    validate_identity(metric, state, source)
    if state['provenance'].get('uses_test_queries') is not False:
        raise ValueError('Expected mapping-only provenance')
    records = torch.load(a.cache, map_location='cpu', weights_only=False)['queries']
    original_count = len(state['anchor_ids'])
    if not a.completion:
        rows = torch.tensor([i for i, kind in enumerate(state['anchor_candidate_kind'])
                             if kind != 'depth_proposed_projective_completion'], dtype=torch.long)
        state = subset(state, rows)
    if not a.fusion:
        state['anchor_features'] = representative_features(state, records)
        state['anchor_observation_features'] = state['anchor_features'].clone()
        state['anchor_descriptor_residual'] = torch.zeros_like(state['anchor_features'])
    state['provenance'] = dict(state['provenance'], observation_factorial=dict(
        fusion=bool(a.fusion), completion=bool(a.completion),
        source_sha256=sha256(source), cache_sha256=sha256(a.cache),
        representative='lexicographically_first_mapping_observation',
        quality_scores='fixed_source_scores_not_reestimated_after_descriptor_change'))
    a.output.mkdir(parents=True)
    def save(folder, revised):
        folder.mkdir()
        path = folder / 'projective_anchor_map.pt'
        torch.save(revised, path)
        identity = folder / 'identity_metric.pt'
        torch.save(dict(metric, landmark_indices=revised['anchor_ids'],
                        map_path=str(path), map_sha256=sha256(path)), identity)
        export_map(path, identity, folder / 'deployment_map.pt')
    save(a.output / 'full', state)
    start = time.monotonic()
    rows, trace = select_compact_rows(state, records)
    seconds = time.monotonic() - start
    revised = subset(state, rows)
    save(a.output / 'compact', revised)
    torch.save(trace, a.output / 'selector_trace.pt')
    (a.output / 'report.json').write_text(json.dumps(dict(
        source_count=original_count, full_count=len(state['anchor_ids']),
        compact_count=len(rows), selection_seconds=seconds,
        retained_source_rows=rows.tolist(), uses_test_queries=False), indent=2))


if __name__ == '__main__':
    main()
