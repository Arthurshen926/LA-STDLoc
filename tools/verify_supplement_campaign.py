"""Check completed experimental controls before timing or paper aggregation."""
from pathlib import Path
import argparse
import numpy as np
import torch
from supplement_experiments import SCENES, read, write


def verify(root):
    protocol = read(root/'protocol.json')
    checks = []
    def load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    for scene in SCENES:
        s = protocol['scenes'][scene]
        reference = read(Path(s['reference'])/'results.json')
        names = [r['image_name'] for r in reference]
        original = load(s['compact_map'])
        assert len(original['anchor_ids']) == s['K']
        for strategy in ['quality', 'spatial']:
            selected = load(root/'selector'/scene/strategy/'deployment_map.pt')
            assert len(selected['anchor_ids']) == s['K']
        maps = [load(root/'geometry'/scene/k/'deployment_map.pt') for k in ['ray', 'depth', 'center']]
        common = np.load(root/'geometry'/scene/'common_observations.npz')
        assert np.all(np.diff(common['offsets']) >= 3)
        for state in maps:
            assert len(state['anchor_ids']) == len(common['source_rows'])
            assert torch.equal(state['anchor_features'], maps[0]['anchor_features'])
            assert torch.equal(state['anchor_ids'], maps[0]['anchor_ids'])
            assert state['provenance']['common_observations_sha256'] == maps[0]['provenance']['common_observations_sha256']
        classic = read(root/'classic'/scene/'fit/summary.json')
        for strategy in ['spatial', 'ours_spatial']:
            state = load(root/'classic'/scene/'fit'/strategy/'deployment_map.pt')
            assert len(state['anchor_ids']) == classic['actual_K']
        assert classic['actual_K'] <= s['K']
        paths = list((root/'evaluations').glob(scene+'_*/results.json'))
        assert len(paths) == 8, (scene, paths)
        for path in paths:
            records = read(path)
            assert [r['image_name'] for r in records] == names
            manifest = read(path.parent/'manifest.json')
            config = manifest['config']
            assert config['confidence'] == (.99999 if manifest['budget'] == 'D' else .999)
        replay = read(root/'ransac'/scene/'replay/D/results.json')
        full = read(root/'evaluations'/(scene+'_native_D')/'results.json')
        for a, b in zip(replay, full):
            assert a['image_name'] == b['image_name'] and a['success'] == b['success']
            assert np.array_equal(a['pose_w2c'], b['pose_w2c']), (scene, a['image_name'])
        checks.append(dict(scene=scene, selector_K=s['K'], geometry_K=len(common['source_rows']),
            classic_K=classic['actual_K'], classic_shortfall=classic['budget_shortfall'],
            query_count=len(names), D_full_pipeline_exact_replay=True))
    write(root/'verification.json', dict(complete=True, checks=checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root)
