"""Audit existing native inference representations and plot deployment costs.

No localization, model training, or representation conversion is run. Baseline
files are the original iteration-30000 inference assets, not working directories.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import pickle
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anygsloc.artifacts import validate

OUT = ROOT / 'paper/experiments_cvpr_20261003'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'docs/paper/evidence_20260917'
BASE = Path('/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915')
MAPS = Path('/mnt/pool/sqy/anygsloc_observation_factorial_20260914/maps')
SCENES = ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']
COLORS = {'STDLoc': '#3676a8', 'ULFLoc': '#85579a', 'AnyGSLoc': '#d87a26'}
INK = '#243349'
torch.set_num_threads(4)

def rows(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

class CPUUnpickler(pickle.Unpickler):
    """Read trusted archived tensors on CPU without allocating GPU storage."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

def read_pickle(path):
    with path.open('rb') as stream:
        return CPUUnpickler(stream).load()

def ply_header(path):
    header = []
    with path.open('rb') as stream:
        while True:
            line = stream.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
    n = int(next(x for x in header if x.startswith('element vertex ')).split()[-1])
    props = [x.split()[-1] for x in header if x.startswith('property ')]
    return {'vertices': n, 'properties': props, 'header_bytes': sum(len(x)+1 for x in header)}

def file_record(path, role):
    return {'path': str(path), 'role': role, 'bytes': path.stat().st_size, 'sha256': sha(path)}

def audit():
    entries = []
    for semantics in ['off', 'on']:
        for method in ['STDLoc', 'ULFLoc']:
            for scene in SCENES:
                folder = BASE / 'baseline_models' / f'{method}_{scene}_mapping_semantics_{semantics}'
                files = [('point_cloud/iteration_30000/point_cloud.ply', 'Gaussian representation'),
                         ('cfg_args', 'Native scene-model configuration')]
                if method == 'STDLoc':
                    files += [('detector/sampled_idx.pkl', 'Sparse landmark indices'),
                              ('detector/30000_detector.pth', 'Scene-specific detector weights')]
                    idx_path = folder / 'detector/sampled_idx.pkl'
                    detector = torch.load(folder / 'detector/30000_detector.pth', map_location='cpu', weights_only=True)
                    extra = {'detector_tensor_shapes': {k: list(v.shape) for k,v in detector.items()}}
                else:
                    files += [('test/keypoints_sampled_idx.pkl', 'Sparse landmark indices'),
                              ('test/keypoints_features.pkl', 'Fused landmark descriptors')]
                    idx_path = folder / 'test/keypoints_sampled_idx.pkl'
                    feat = torch.as_tensor(read_pickle(folder / 'test/keypoints_features.pkl')).cpu()
                    extra = {'descriptor_shape': list(feat.shape), 'descriptor_finite': bool(torch.isfinite(feat).all())}
                indices = torch.as_tensor(read_pickle(idx_path)).cpu().numpy().reshape(-1)
                header = ply_header(folder / files[0][0])
                assert len(indices) > 0 and indices.min() >= 0 and indices.max() < header['vertices']
                assert all(k in header['properties'] for k in ['x','y','z','opacity','scale_0','rot_0'])
                if method == 'ULFLoc':
                    assert feat.shape[0] == len(indices) and extra['descriptor_finite']
                else:
                    assert sum(p.startswith('loc_') for p in header['properties']) == 256
                assets = [file_record(folder / p, role) for p, role in files]
                entries.append(dict(method=method, scene=scene, mapping_semantics=semantics,
                    stage='final', files=assets, total_bytes=sum(f['bytes'] for f in assets),
                    selected_landmarks=len(indices), ply=header, **extra))
                print(f'audited {method} {scene} masks={semantics}', flush=True)
    for selection in ['compact', 'full']:
        for scene in SCENES:
            path = MAPS / f'{scene}_v1_f1_c1' / selection / 'deployment_map.pt'
            state = validate(torch.load(path, map_location='cpu', weights_only=True))
            assets = [file_record(path, 'XYZ + frozen descriptors + IDs + provenance')]
            entries.append(dict(method='AnyGSLoc', scene=scene, mapping_semantics='prior-dependent',
                stage=selection, files=assets, total_bytes=assets[0]['bytes'], selected_landmarks=len(state['anchor_ids'])))
    refs = [BASE/'baselines/STDLoc/stdloc.py', BASE/'baselines/ULFLoc/ulfloc.py',
            BASE/'baselines/STDLoc/scene/__init__.py', BASE/'baselines/ULFLoc/scene/__init__.py',
            BASE/'baselines/STDLoc/scene/gaussian_model.py', BASE/'baselines/ULFLoc/scene/gaussian_model.py',
            BASE/'code/tools/baseline_protocol_v2.py', ROOT/'anygsloc/online/runtime.py', ROOT/'anygsloc/artifacts.py']
    weights = []
    for method in ['STDLoc','ULFLoc']:
        path = BASE / 'baselines' / method / 'encoders/sp_encoder/weights/superpoint_v1.pth'
        if path.is_file():
            weights.append(file_record(path, 'Shared SuperPoint weights; excluded from scene storage'))
    report = dict(audit_date='2026-10-03', entries=entries,
        storage_contract={
            'metric': 'Sum of original native inference representation files across five Cambridge scenes, decimal MB.',
            'baseline_configuration': 'Final stage, iteration 30000, matching the archived pose evaluation; main figure uses mapping semantics off.',
            'included': 'Full native final PLY, one index file, STDLoc scene detector or ULF-Loc landmark feature file, baseline cfg_args; original AnyGSLoc deployment_map.pt including provenance.',
            'excluded': 'Intermediate checkpoints, training logs, input SfM cloud, training images, pose/evaluation results, duplicate benchmark copies, shared code and pretrained weights, query RGB and query calibration/rectification metadata.',
            'query_contract': 'Query RGB plus camera calibration and geometric rectification support are inputs common to all methods; no query semantic masks. Query records and evaluation ground truth are not scene representations.',
            'representation_preserved': 'No precision changes, property stripping, tensor repacking, pruning, quantization, or retraining. Native file overhead and unused serialized properties remain included.',
            'qualification': 'Source-code dependency audit plus file/hash/schema/range checks. Not a lower bound on theoretically minimal storage and not an isolated deployment execution test.',
        }, code_sources=[file_record(p, 'Inference dependency evidence') for p in refs],
        shared_weights_excluded=weights)
    (OUT/'E5_storage_audit.json').write_text(json.dumps(report, indent=2))
    with (OUT/'E5_storage_by_scene.csv').open('w') as f:
        keys=['method','scene','mapping_semantics','stage','storage_MB','selected_landmarks']
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader()
        for e in entries:
            writer.writerow({**{k:e[k] for k in keys if k!='storage_MB'},'storage_MB':e['total_bytes']/1e6})
    return report

def build(report):
    accuracy, timing, factorial = rows('baseline_macro.csv'), rows('baseline_timing_macro.csv'), rows('factorial_macro.csv')
    compact=next(r for r in factorial if r['selection']=='compact' and all(r[k]=='1' for k in ['V2','fusion','completion']))
    points=[]
    for method in ['STDLoc','ULFLoc']:
        t=next(r for r in timing if r['method']==method and r['mapping_mask']=='off')
        storage=sum(e['total_bytes'] for e in report['entries'] if e['method']==method and e['mapping_semantics']=='off')/1e6
        for stage,key in [('coarse','coarse_prefix_ms'),('final','coarse_plus_fine_ms')]:
            a=next(r for r in accuracy if r['method']==method and r['mapping_mask']=='off' and r['stage']==('fine' if stage=='final' else stage))
            points.append(dict(method=method,stage=stage,latency_ms=float(t[key]),storage_MB=storage if stage=='final' else None,
                **{k:float(a[k]) for k in ['TE_cm','RE_deg','R5_pct']}))
    storage=sum(e['total_bytes'] for e in report['entries'] if e['method']=='AnyGSLoc' and e['stage']=='compact')/1e6
    assert np.isclose(storage,float(compact['total_map_MB']),rtol=0,atol=1e-9)
    points.append(dict(method='AnyGSLoc',stage='single',latency_ms=float(compact['core_mean_ms']),storage_MB=storage,
        **{k:float(compact[k]) for k in ['TE_cm','RE_deg','R5_pct']}))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,
        'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#89939e'})
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.15),sharey=True)
    fig.subplots_adjust(left=.09,right=.975,bottom=.21,top=.88,wspace=.14)
    for i,ax in enumerate(axs):
        ax.set_xscale('log');ax.set_ylim(0,52);ax.set_yticks([0,10,20,30,40,50]);ax.grid(axis='y',alpha=.15,lw=.5)
        ax.set_axisbelow(True);ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='both',labelsize=7,pad=2,length=3)
        ax.set_title(['(a) Online computation','(b) Online representation'][i],loc='left',pad=8)
    axs[0].set(xlim=(47,2100),xticks=[50,100,300,1000],xlabel='Mean pipeline latency (ms, log)',ylabel='Recall @ 5 cm / 5° (%) ↑')
    axs[1].set(xlim=(125,3650),xticks=[200,500,1000,3000],xlabel='Scene-specific storage (MB, log)')
    runtime_offsets={('AnyGSLoc','single'):(-16,11),('STDLoc','coarse'):(-5,-10),('STDLoc','final'):(-7,11),
                     ('ULFLoc','coarse'):(8,-12),('ULFLoc','final'):(-2,12)}
    storage_offsets={'AnyGSLoc':(5,10),'STDLoc':(-5,-10),'ULFLoc':(-7,-10)}
    for p in points:
        col=COLORS[p['method']];marker='*' if p['method']=='AnyGSLoc' else 'o'
        open_marker=p['stage']=='coarse'; label=p['method'].replace('ULFLoc','ULF-Loc')
        for i,key in enumerate(['latency_ms','storage_MB']):
            if p[key] is None: continue
            ax=axs[i];x=p[key];y=p['R5_pct']
            ax.scatter(x,y,s=85 if marker=='*' else 33,marker=marker,facecolor='white' if open_marker else col,
                       edgecolor=col,linewidth=1.2,zorder=4)
            offset=runtime_offsets[(p['method'],p['stage'])] if i==0 else storage_offsets[p['method']]
            align='left' if p['method']=='AnyGSLoc' else ('right' if offset[0]<0 else 'left')
            name=label+(' C' if open_marker else (' F' if p['stage']=='final' and i==0 else ''))
            cost=f"{x:,.1f} {'ms' if i==0 else 'MB'}"
            text=f"{name}\n{cost}\n{p['TE_cm']:.2f} cm / {p['RE_deg']:.3f}°"
            ax.annotate(text,(x,y),xytext=offset,textcoords='offset points',ha=align,va='bottom' if offset[1]>0 else 'top',
                fontsize=7,linespacing=1.25,color=col,clip_on=False)
    for method in ['STDLoc','ULFLoc']:
        p,q=[p for p in points if p['method']==method]
        axs[0].annotate('',(q['latency_ms'],q['R5_pct']),(p['latency_ms'],p['R5_pct']),
            arrowprops=dict(arrowstyle='->',color=COLORS[method],lw=1,shrinkA=4,shrinkB=5),zorder=2)
    fig.text(.09,.035,'C: coarse     F: cumulative final     ★: single stage     Labels: TE / RE',fontsize=7.2)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'E5_deployment.{ext}',dpi=320,facecolor='white')
    plt.close(fig)
    caption=('Accuracy versus online deployment cost on the five Cambridge Landmarks scenes. '
        '(a) Mean pipeline latency versus recall at 5 cm / 5 degrees; arrows connect the coarse and cumulative final stages of each reproduced baseline. '
        'Coarse time is the native pipeline prefix through coarse-pose estimation, including shared query precomputation, not an isolated sparse-only module. '
        '(b) Final-stage accuracy versus the sum of native scene-specific inference representation files across the same five scenes, in decimal MB. '
        'Point labels report the mean of scene-median translation and rotation errors, not pooled medians. '
        'All measurements use the archived native method configurations with query semantic masking disabled; baseline mapping semantics are off. '
        'Timing uses the existing warmed benchmarks and excludes outer disk I/O and pose-error evaluation; the native ULF-Loc coarse inlier-rate diagnostic against the reference camera remains inside its timed pipeline. Native resolutions and configurations differ. '
        'Storage includes final Gaussian models, required landmark assets and scene-specific detector weights where applicable; it excludes shared pretrained weights, intermediate checkpoints, logs, duplicate benchmark assets, and query RGB/calibration/rectification inputs. '
        'Native representations are preserved without compression or repacking. The storage audit follows source-level inference dependencies and verifies existing files; it does not establish an optimally compressed lower bound or rerun deployment. '
        'The AnyGSLoc point uses the compact map. Full/compact selection effects are reported separately.')
    (OUT/'E5_deployment.caption.txt').write_text(caption+'\n')
    fact=rows('factorial_80.csv')
    tails=[{k:r[k] for k in ['scene','core_mean_ms','core_p95_ms_round_mean','loop_fps_round_mean']} for r in fact
           if r['selection']=='compact' and all(r[k]=='1' for k in ['v2','fusion','completion'])]
    manifest=dict(figure='E5_deployment',points=points,caption=caption,anygsloc_tail_latency=tails,
        sources=[file_record(DATA/name,'Archived aggregate table') for name in ['baseline_macro.csv','baseline_timing_macro.csv','factorial_macro.csv','factorial_80.csv']],
        storage_audit='E5_storage_audit.json',limitations=['No equal-resolution speedup claim.','No real-time or tail-latency claim.','No baseline coarse-storage point; final-stage assets only.',
            'Native ULF-Loc reference-camera inlier-rate diagnostic is retained in timing; this is not a purified inference-only comparison.',
            'Coarse latency includes the native shared query precomputation prefix.'])
    (OUT/'E5_deployment.json').write_text(json.dumps(manifest,indent=2))
    notes=['Deployment representation audit — 2026-10-03','',json.dumps(report['storage_contract'],ensure_ascii=False,indent=2),'',
           'Main comparison (mapping semantics off; five-scene sum):']
    for p in points:
        if p['storage_MB'] is not None:
            notes.append(f"{p['method']}: {p['storage_MB']:.6f} MB; TE {p['TE_cm']:.4f} cm; RE {p['RE_deg']:.6f} deg; R5 {p['R5_pct']:.4f}%; mean {p['latency_ms']:.4f} ms")
    notes += ['', 'See E5_storage_by_scene.csv for both baseline mapping-mask settings and AnyGSLoc full/compact.',
              'Shared SuperPoint checkpoint bytes are recorded separately in E5_storage_audit.json.',
              'Native loaders: STDLoc/stdloc.py lines 121–139, ULFLoc/ulfloc.py lines 65–82; Scene loads the iteration PLY; AnyGSLoc Localizer loads deployment_map.pt.',
              'The old supplied_assets_bytes directory totals are deliberately not used as the deployment metric.']
    (OUT/'E5_storage_audit.txt').write_text('\n'.join(notes)+'\n')
    print(json.dumps(points,indent=2))

if __name__=='__main__':
    audit_file=OUT/'E5_storage_audit.json'
    report=json.loads(audit_file.read_text()) if '--plot-only' in sys.argv else audit()
    build(report)
