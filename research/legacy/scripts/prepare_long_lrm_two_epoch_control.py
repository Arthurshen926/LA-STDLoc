"""Supplemental optimization-budget diagnostic; never replace the +1k matrix."""
import json
import shutil
from pathlib import Path
from common.hashing import sha256_file


def main():
    old=Path('/mnt/pool/sqy/anygsloc_three_gpu_20260910')
    root=Path('/mnt/pool/sqy/anygsloc_long_lrm_two_epoch_control_20260910');root.mkdir(exist_ok=False)
    snap=root/'code';shutil.copytree(old/'code',snap,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    helper=snap/'scripts/wait_queue_lane_finished.py';shutil.copyfile(Path(__file__).with_name(helper.name),helper)
    specs=[]
    for gpu,scene,count in [(0,'GreatCourt',1531),(1,'StMarysChurch',1487)]:
        iterations=2*count
        original=json.loads((old/f'gpu{gpu}/jobs.json').read_text())
        selected=[j for j in original if j['id'] in [f'{scene}_long_lrm_ft1000',f'{scene}_long_lrm_1k_map',f'{scene}_long_lrm_1k_eval2026',f'{scene}_long_lrm_1k_eval2027']]
        if len(selected)!=4:raise ValueError('missing source jobs')
        ids={j['id'] for j in selected}
        wait={'id':'wait_preceding_lane','command':['/root/miniconda3/envs/g4splat/bin/python','-m','scripts.wait_queue_lane_finished','--state',f'/mnt/pool/sqy/anygsloc_invalid_primitive_recovery_20260910/gpu{gpu}/state.json'],
              'deps':[],'outputs':[],'cwd':str(snap),'env':{'PYTHONPATH':str(snap),'CUDA_VISIBLE_DEVICES':str(gpu)}}
        for j in selected:
            j['deps']=[d for d in j['deps'] if d in ids] or [wait['id']]
            encoded=json.dumps(j).replace(str(old/'code'),str(snap)).replace(str(old/'runs'),str(root/'runs'))
            encoded=encoded.replace(str(old/f'priors/{scene}/long_lrm/ft1000'),str(root/f'priors/{scene}/long_lrm/ft_two_epochs'))
            encoded=encoded.replace(f'{scene}_long_lrm_ft1000',f'{scene}_long_lrm_ft_two_epochs').replace(f'{scene}_long_lrm_1k',f'{scene}_long_lrm_two_epochs')
            encoded=encoded.replace('/iteration_1000/',f'/iteration_{iterations}/')
            j.clear();j.update(json.loads(encoded))
            c=j['command']
            if '--iterations' in c:
                for flag in ['--iterations','--position_lr_max_steps','--save_iterations']:c[c.index(flag)+1]=str(iterations)
                c[c.index('--test_iterations')+1]=str(iterations+1)
        lane=root/f'gpu{gpu}';lane.mkdir();(lane/'jobs.json').write_text(json.dumps([wait]+selected,indent=2))
        specs.append({'scene':scene,'mapping_count':count,'iterations':iterations,'condition':'two full mapping-view sweeps, no densification, original black background held fixed',
                      'comparison':'supplemental budget control, not the fixed +1k comparison'})
    (root/'protocol.json').write_text(json.dumps(specs,indent=2))
    (root/'code_manifest.json').write_text(json.dumps({str(p.relative_to(snap)):sha256_file(p) for p in snap.rglob('*') if p.is_file()},indent=2))
    for p in snap.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print(root,flush=True)


if __name__=='__main__':main()
