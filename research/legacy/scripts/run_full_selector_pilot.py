"""Mapping-only full selector pilot, with staged and equal-size controls."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from topology.sufficiency_selector import HierarchicalSufficiencySelector
from topology.adaptive_distillation import _build_pose_evidence, _initial_pose_state
from topology.dynamic_reserve import spatial_voxel_ids
from scripts.materialize_purity_candidates import subset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scene', default='ShopFacade')
    p.add_argument('--seeds', type=int, nargs='+', default=[2026])
    p.add_argument('--exclude-candidate-completion', action='store_true')
    p.add_argument('--input-map-dir', type=Path)
    p.add_argument(
        '--source-scene-root',
        type=Path,
        help=(
            'Mapping-only scene artifact root containing observations/. '
            'Defaults to the historical V24 scene root.'
        ),
    )
    a = p.parse_args(); torch.set_num_threads(4)
    started = time.perf_counter()
    a.root.mkdir(parents=True, exist_ok=False)
    source = (
        a.source_scene_root.expanduser().resolve()
        if a.source_scene_root is not None
        else Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901') / a.scene
    )
    map_dir = a.input_map_dir or source/'projective_map'
    state = torch.load(map_dir/'projective_anchor_map.pt', map_location='cpu', weights_only=False)
    metric = torch.load(map_dir/'identity_metric.pt', map_location='cpu', weights_only=False)
    source_count = len(state['anchor_ids'])
    if a.exclude_candidate_completion:
        keep = torch.tensor([i for i, kind in enumerate(state['anchor_candidate_kind'])
                             if kind != 'depth_proposed_projective_completion'])
        state = subset(state, keep)
        state['provenance'] = dict(state['provenance'], candidate_completion_excluded=True,
                                  retained_F0_rows=keep)
    cache = torch.load(source/'observations/rendered_rgb_feature_cache.pt', map_location='cpu', weights_only=False)['queries']
    assert state['provenance']['uses_test_queries'] is False
    names = state['v6_mapping_query_names']; n = len(state['anchor_ids']); q = len(names)
    csr = state['projective_anchor_observations']; off = csr['observation_offsets'].tolist()
    queries = csr['query_indices'].tolist(); points = csr['keypoint_indices'].tolist()
    edges = []
    for i in range(n):
        edge = {}
        for j in range(off[i],off[i+1]): edge.setdefault(queries[j],[]).append(points[j])
        edges.append(edge)
    quality = state['anchor_matchability'].float()
    order = torch.tensor(sorted(range(n), key=lambda i: (-float(quality[i]),-len(edges[i]),i)))
    selector = HierarchicalSufficiencySelector(edges,q)
    core, _ = selector.select_precision(order,256,minimum_count=1024,maximum_count=n,check_interval=512)
    variants = {'core': core.sort().values}
    mask = torch.ones(n,dtype=torch.bool); mask[core]=False
    _, matching, _ = selector.complete_matching(order[mask[order]],quality,state['v6_mapping_query_bins'],
        requested_rows_per_query=256, maximum_reserve=n-len(core))
    selected = selector.selected_ids
    variants['matching'] = selected.sort().values
    print('core/matching',len(core),len(selected),flush=True)
    voxels = spatial_voxel_ids(state['anchor_xyz'],0.5)
    evidence = _build_pose_evidence(edges,state['anchor_xyz'],quality,state['anchor_position_covariance'],names,cache,voxels,
        pixel_variance=4., task_translation_m=.05,task_rotation_deg=5.)
    info,rows,cells,depths,spatial,assignments = _initial_pose_state(selected,matching,evidence,q)
    mask[selected]=False
    selector.complete_observability(evidence,info,rows,cells,depths,spatial,order[mask[order]],
        torch.arange(n),voxels,maximum_additions=min(4096,n-len(selected)),
        maximum_per_source=1,maximum_per_voxel=3,minimum_relative_gain=.02,
        initial_assignments=assignments)
    variants['full_selector'] = selector.selected_ids.sort().values
    count = len(variants['full_selector'])-len(selected)
    variants['equal_size_quality'] = torch.cat([selected,order[mask[order]][:count]]).sort().values
    torch.save(selector.artifact(),a.root/'selector_trace.pt')
    original=json.loads(Path('/mnt/pool/sqy/anygsloc_core65_budget_20260910/gpu0/jobs.json').read_text())
    ref=next(j for j in original if j['id']==f'{a.scene}_core65_min250_conf999_max30k_seed2026')
    jobs=[]
    for variant,indices in variants.items():
        output=a.root/'maps'/variant;output.mkdir(parents=True)
        revised=subset(state,indices)
        revised['provenance']=dict(state['provenance'],full_selector_pilot=dict(variant=variant,uses_test_queries=False,retained_source_rows=indices))
        path=output/'projective_anchor_map.pt';torch.save(revised,path)
        with path.open('rb') as f: digest=hashlib.sha256(f.read()).hexdigest()
        torch.save(dict(metric,landmark_indices=revised['anchor_ids'],map_path=str(path),map_sha256=digest),output/'identity_metric.pt')
        command=list(ref['command']);result=a.root/'runs'/variant
        for flag,value in [('--map',path),('--metric-state',output/'identity_metric.pt'),('--output',result)]:command[command.index(flag)+1]=str(value)
        jobs.append(dict(ref,id=variant,command=command,outputs=[str(result/'summary.json'),str(result/'results.json')]))
    command=list(ref['command']);result=a.root/'runs/full_map';command[command.index('--output')+1]=str(result)
    for flag,name in [('--map','projective_anchor_map.pt'),('--metric-state','identity_metric.pt')]:
        command[command.index(flag)+1]=str(map_dir/name)
    jobs.append(dict(ref,id='full_map',command=command,outputs=[str(result/'summary.json'),str(result/'results.json')]))
    if a.exclude_candidate_completion:
        # The historical full_map command still points to the original F0; do not
        # mislabel it as the completion-free input. This run tests selector outputs.
        jobs = [job for job in jobs if job['id'] != 'full_map']
    expanded = []
    for seed in a.seeds:
        for job in jobs:
            command = list(job['command'])
            command[command.index('--seed')+1] = str(seed)
            result = a.root/'runs'/f"{job['id']}_seed{seed}"
            command[command.index('--output')+1] = str(result)
            expanded.append(dict(job,id=f"{job['id']}_seed{seed}",command=command,
                outputs=[str(result/'summary.json'),str(result/'results.json')]))
    (a.root/'jobs.json').write_text(json.dumps(expanded,indent=2))
    (a.root/'protocol.json').write_text(json.dumps(dict(scene=a.scene,mapping_only=True,offline_feedback=False,
        target_rows=256,pose_addition_cap=4096,voxel_m=.5,variants={k:len(v) for k,v in variants.items()},
        online='identical frozen Fast including 65% gate',source=str(source),seeds=a.seeds,
        build_seconds=time.perf_counter()-started,
        candidate_completion_excluded=a.exclude_candidate_completion,
        source_anchor_count=source_count, selector_input_anchor_count=n,
        caveat='Pilot parameters; finite pose budget, no global sufficiency guarantee'),indent=2))
    print('ready', {k:len(v) for k,v in variants.items()},flush=True)


if __name__=='__main__': main()
