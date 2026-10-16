"""Auditable, figure-only Gaussian-prior comparison from frozen experiments.

prepare/draw: i3dgs environment; render: g4splat environment. No evaluation,
training, map construction, or input mutation is performed.
"""
from pathlib import Path
import argparse, csv, gc, hashlib, json, struct, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POOL = Path('/mnt/pool/sqy')
OUT = ROOT / 'paper/experiments_cvpr_20261003'
ASSETS = OUT / 'assets/e4'
PROV = OUT / 'E4_gaussian_prior.provenance.json'
KEYS = ['matcha', 'vanilla_2dgs', 'vanilla_3dgs', 'i3dgs_raw', 'i3dgs_1k']
LABELS = ['MAtCHA', '2DGS', '3DGS', 'i3DGS raw', 'i3DGS +1k']
SCENE = 'StMarysChurch'
GT_COLOR, EST_COLOR = '#219653', '#cc2f83'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def project(xyz, pose, K):
    v = np.asarray(xyz) @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]
    p = v @ np.asarray(K).T
    return p[:, :2] / np.maximum(p[:, 2:], 1e-9), v[:, 2]


def read_sfm(path):
    rows = []
    with open(path, 'rb') as f:
        count, = struct.unpack('<Q', f.read(8))
        for _ in range(count):
            row = struct.unpack('<QdddBBBd', f.read(43))
            track, = struct.unpack('<Q', f.read(8))
            f.seek(8 * track, 1)
            rows.append(row)
    return np.array([x[0] for x in rows]), np.array([x[1:4] for x in rows]), np.array([x[-1] for x in rows])


def prepare():
    import torch
    from scipy.stats import rankdata
    from scipy.ndimage import sobel
    from scipy.spatial import cKDTree
    from PIL import Image
    from anygsloc.data.datasets import ColmapDataset
    torch.set_num_threads(4)
    ASSETS.mkdir(parents=True, exist_ok=True)
    suite = POOL / 'anygsloc_cambridge_suite_v2_20260914'
    primary = POOL / 'anygsloc_protocol_v2_primary_20260914'
    jobs = read(suite / 'gpu0/state.json')['jobs']
    factorial = POOL/'anygsloc_observation_factorial_20260914'
    matcha_cell = next(x for x in read(factorial/'registry.json') if x['id']==SCENE+'_v1_f1_c1_compact')
    p = {'scene': SCENE, 'keys': KEYS, 'labels': LABELS, 'sources': {}, 'priors': {},
         'scope': 'Existing frozen localization outputs; offline visualization only. No new algorithm experiments.',
         'selection': {'scene_rule': 'StMarysChurch fixed before comparing individual query errors: structure-rich scene recommended in design brief.',
                       'query_rule': 'All five priors valid; e=max(TE_cm/5, RE_deg/5); percentile rank within their common valid queries; choose mean rank closest to 0.5, lexical tie break.',
                       'normalized_error': 'max(TE_cm/5, RE_deg/5)'},
         'limitations': ['Input training/preparation conditions differ across Gaussian producers.',
                        'SfM overlays show common 3D points, not recovered surface edges or ground-truth surfaces.',
                        'PCA colors encode descriptor-space directions, not semantic classes.',
                        'A single illustrative query does not represent all possible failure modes.']}
    runs, records = {}, {}
    for key, label in zip(KEYS, LABELS):
        run = Path(matcha_cell['result']) if key == 'matcha' else suite / 'runs' / f'prior_{SCENE}_{key}'
        command = None if key == 'matcha' else jobs[f'prior_{SCENE}_{key}']['command']
        mp = Path(matcha_cell['map']) if key=='matcha' else Path(command[command.index('--map') + 1])
        manifest = read(run / 'manifest.json')
        actual = digest(mp)
        assert actual == manifest['map_sha256'], (key, 'map differs from evaluated map')
        results = read(run / 'results.json')
        records[key] = {r['image_name']: r for r in results if r['success'] and r.get('pose_w2c') is not None and np.isfinite(r['translation_error_cm']) and np.isfinite(r['rotation_error_deg'])}
        runs[key] = run
        p['priors'][key] = {'label': label, 'map': str(mp), 'map_sha256': actual,
                            'results': str(run / 'results.json'), 'results_sha256': digest(run / 'results.json')}
    names = sorted(set.intersection(*[set(r) for r in records.values()]))
    errors = np.array([[max(records[key][n]['translation_error_cm']/5, records[key][n]['rotation_error_deg']/5) for n in names] for key in KEYS])
    ranks = np.array([(rankdata(x, method='average')-1)/(len(names)-1) for x in errors])
    scores = ranks.mean(0)
    chosen = min(range(len(names)), key=lambda i: (abs(scores[i]-.5), names[i]))
    name = names[chosen]
    p['selection'].update(query=name, common_valid_query_count=len(names), mean_percentile_rank=float(scores[chosen]), per_prior_percentile_rank=dict(zip(KEYS, ranks[:, chosen].tolist())))
    with (ASSETS/'query_selection.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['query','mean_rank']+[k+'_rank' for k in KEYS])
        for i, n in enumerate(names): writer.writerow([n, scores[i], *ranks[:, i]])
    dataset = ColmapDataset(POOL/'Cambridge_stdloc'/SCENE, load_masks=False)
    camera = dataset.camera(name)
    K = np.array([[camera.width/(2*np.tan(camera.fov_x/2)),0,camera.width/2], [0,camera.height/(2*np.tan(camera.fov_y/2)),camera.height/2], [0,0,1.]])
    p['camera'] = {'query_path': str(camera.image_path), 'query_sha256': digest(camera.image_path), 'width':camera.width, 'height':camera.height, 'K':K.tolist(), 'gt_pose_w2c':camera.pose_w2c.tolist(),
                   'calibration': 'Frozen evaluation ColmapDataset, processed rectified RGB, OpenCV same-K contract; world-to-camera coordinates.'}
    Image.open(camera.image_path).convert('RGB').save(ASSETS/'real_query.png')
    feat, xyz = {}, {}
    for key in KEYS:
        entry = p['priors'][key]
        d = torch.load(entry['map'], map_location='cpu', weights_only=False, mmap=True)
        xyz[key] = d['anchor_xyz'].numpy().copy()
        feat[key] = d['anchor_features'].numpy().copy()
        assert np.isfinite(xyz[key]).all() and np.isfinite(feat[key]).all()
        entry.update(landmarks=len(xyz[key]), result=records[key][name])
        r = entry['result']; estimated = np.array(r['pose_w2c']); gt = camera.pose_w2c.astype(float)
        te = np.linalg.norm(np.linalg.inv(estimated)[:3,3]-np.linalg.inv(gt)[:3,3])*100
        # SVD orthonormalization controls FP32 roundoff for the tiny rotations.
        def ortho(a):
            u, _, v = np.linalg.svd(a); return u@v
        rot = ortho(estimated[:3,:3]) @ ortho(gt[:3,:3]).T
        re = np.degrees(np.arccos(np.clip((np.trace(rot)-1)/2,-1,1)))
        assert abs(te-r['translation_error_cm']) < .002
        assert abs(re-r['rotation_error_deg']) < .005
        entry['pose_error_recomputed'] = {'TE_cm':float(te), 'RE_deg':float(re)}
        if key == 'matcha':
            v = next(x for x in read(ROOT/'paper/figures_v5/selection.json')['views'] if x['scene']==SCENE)
            entry.update(ply=v['prior'], ply_sha256=v['prior_sha256'], gaussian_type='2dgs', sh_degree=3)
        else:
            audit = next((POOL/f'anygsloc_three_gpu_20260910/runs/{SCENE}_{key}/v2_audit').glob('*.pt'))
            binding = torch.load(audit,map_location='cpu',weights_only=False,mmap=True)['input']
            entry.update(ply=binding['gaussian_ply'], ply_sha256=binding['gaussian_ply_sha256'], gaussian_type=binding['gaussian_type'], sh_degree=binding['sh_degree'], audit=str(audit))
        print('loaded', key, len(xyz[key]), flush=True)
    n = min(8192, min(len(x) for x in feat.values()))
    rng = np.random.default_rng(2026)
    samples = {}; sample_ids = {}
    for key in KEYS:
        sample_ids[key] = rng.choice(len(feat[key]),n,replace=False)
        samples[key] = feat[key][sample_ids[key]]
    shared = np.concatenate(list(samples.values())).astype(np.float64)
    mean = shared.mean(0)
    vals, vecs = np.linalg.eigh((shared-mean).T@(shared-mean)/(len(shared)-1))
    comp = vecs[:,-3:][:,::-1]
    # Deterministic sign: largest absolute loading positive.
    comp *= np.sign(comp[np.argmax(np.abs(comp),axis=0),np.arange(3)])[None]
    rgb_lo, rgb_hi = np.percentile((shared-mean)@comp,[1,99],axis=0)
    np.savez_compressed(ASSETS/'shared_pca.npz', mean=mean, components=comp, low=rgb_lo, high=rgb_hi, **{k+'_sample_ids':v for k,v in sample_ids.items()})
    p['pca'] = {'fit':'one covariance PCA over equal-sized random samples from each evaluated compact map', 'seed':2026,'samples_per_prior':n,'descriptor_dimension':256,'normalization':'Shared pooled 1st–99th percentile per component; clipped to [0,1].','explained_variance_ratio':(vals[-3:][::-1]/vals.sum()).tolist(),'sign_convention':'Largest absolute loading positive'}
    for key in KEYS:
        colors = np.clip(((feat[key]-mean)@comp-rgb_lo)/(rgb_hi-rgb_lo),0,1).astype(np.float32)
        uv, z = project(xyz[key],camera.pose_w2c,K)
        visible = (z>0)&(uv[:,0]>=0)&(uv[:,0]<camera.width)&(uv[:,1]>=0)&(uv[:,1]<camera.height)
        np.savez_compressed(ASSETS/(key+'_map.npz'), uv=uv[visible], depth=z[visible], colors=colors[visible])
        p['priors'][key]['landmarks_in_frustum'] = int(visible.sum())
    sfm = POOL/'Cambridge_stdloc'/SCENE/'sparse/0/points3D.bin'
    ids, points, sfm_error = read_sfm(sfm)
    uv, z = project(points,camera.pose_w2c,K)
    # Point identities selected exclusively from the GT camera and RGB, never estimates.
    eligible = np.where((z>0)&np.isfinite(sfm_error)&(uv[:,0]>48)&(uv[:,0]<camera.width-48)&(uv[:,1]>48)&(uv[:,1]<camera.height-48))[0]
    density=cKDTree(uv[eligible]).query_ball_point(uv[eligible],r=16,return_length=True)
    gray=np.asarray(Image.open(camera.image_path).convert('RGB')).astype(float).mean(2)/255
    gradient=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1))
    coords=np.round(uv[eligible]).astype(int)
    texture=gradient[coords[:,1],coords[:,0]]
    quality=np.zeros(len(points));quality[eligible]=texture*np.log1p(density)
    eligible=eligible[(density>=12)&(texture>=.10)]
    bins = {}
    for i in eligible:
        cell = (int(uv[i,0]//100),int(uv[i,1]//100))
        if cell not in bins or (quality[i],-int(ids[i]))>(quality[bins[cell]],-int(ids[bins[cell]])):
            bins[cell] = i
    selected = np.array([bins[k] for k in sorted(bins)])
    np.savez_compressed(ASSETS/'neutral_geometry.npz', ids=ids[selected], xyz=points[selected], gt_uv=uv[selected])
    # Fixed central/right facade ROI, selected from GT support density only.
    cw,ch=640,360
    rois=[]
    for y in range(120,camera.height-ch-60,60):
        for x in range(400,camera.width-cw-60,80):
            inside=(uv[selected,0]>=x)&(uv[selected,0]<x+cw)&(uv[selected,1]>=y)&(uv[selected,1]<y+ch)
            rois.append((int(inside.sum()),-abs(x+cw/2-camera.width*.65),-y,x,y))
    _,_,_,x,y=max(rois)
    p['neutral_geometry']={'source':str(sfm),'source_sha256':digest(sfm),'point_count':len(selected),'selection':'GT-only RGB gradient >=0.10 and at least 12 projected neighbors in 16px; maximum gradient*log(1+density) per 100x100 pixel cell, tie by point ID. All priors project identical point IDs. SfM export error is not interpreted as confidence.','roi':[x,y,x+cw,y+ch],'roi_rule':'Max GT-selected point count among fixed 640x360 facade windows; ties prefer center near 65% image width then upper window.'}
    prior_macro = list(csv.DictReader((ROOT/'docs/paper/evidence_20260917/prior_macro.csv').open()))
    factorial_rows = list(csv.DictReader((ROOT/'docs/paper/evidence_20260917/factorial_macro.csv').open()))
    for key in KEYS:
        if key=='matcha':
            row=next(r for r in factorial_rows if (r['V2'],r['fusion'],r['completion'],r['selection'])==('1','1','1','compact'))
            agg={k:float(row[k]) for k in ['TE_cm','RE_deg','R5_pct']}
        else:
            row=next(r for r in prior_macro if r['prior']==key)
            agg={k:float(row[k]) for k in ['TE_cm','RE_deg','R5_pct']}
        p['priors'][key]['aggregate']=agg
    p['aggregate_definition']='Unweighted mean of five scene medians (TE, RE) and of five scene R5,5 recalls; not pooled medians.'
    p['matcha_version']='Factorial v1_f1_c1_compact, consistent with E1/E3/E5 compact operating point; other priors use archived compact maps from prior suite.'
    for path in [ROOT/'docs/paper/evidence_20260917/prior_macro.csv',ROOT/'docs/paper/evidence_20260917/factorial_macro.csv']:
        p['sources'][str(path)]=digest(path)
    p['limitations'].append('Sparse SfM projections are GT-frustum/texture/density filtered for display; no mesh-based occlusion visibility is claimed.')
    if PROV.exists():
        old=read(PROV)
        if old['selection']['query']==name:
            for key in KEYS:
                if old['priors'][key].get('render') and old['priors'][key]['ply_sha256']==p['priors'][key]['ply_sha256']:
                    p['priors'][key]['render']=old['priors'][key]['render']
    PROV.write_text(json.dumps(p,indent=2))
    print('chosen',name,'rank',scores[chosen],flush=True)


def render():
    import torch
    from PIL import Image
    from gsplat import rasterization, rasterization_2dgs
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from anygsloc.priors.models import FrozenGaussianModel
    torch.set_num_threads(4)
    p=read(PROV); width=640; height=round(width*p['camera']['height']/p['camera']['width'])
    calibrated=np.array(p['camera']['K'],dtype=np.float64)
    calibrated[:2]*=width/p['camera']['width']
    K=torch.tensor(calibrated,dtype=torch.float32)
    pose=torch.tensor(p['camera']['gt_pose_w2c'],dtype=torch.float32)
    campos=torch.linalg.inv(pose)[:3,3]
    for key in KEYS:
        dest=ASSETS/(key+'_prior.png'); entry=p['priors'][key]
        if dest.exists() and entry.get('render',{}).get('version')==2:continue
        print('render',key,flush=True)
        assert digest(entry['ply'])==entry['ply_sha256']
        # Keep high-dimensional SH on CPU; exactly evaluate at this camera once.
        # All original primitives remain in the native rasterizer, with no culling,
        # opacity threshold, color edit, or subsampling added by this script.
        model=FrozenGaussianModel(entry['sh_degree'],device='cpu')
        model.load_ply(entry['ply'],loc_feature_dim=0)
        rgb=[]
        with torch.inference_mode():
            for start in range(0,len(model.get_xyz),65536):
                stop=start+65536
                coeff=torch.cat([model._features_dc[start:stop],model._features_rest[start:stop]],dim=1)
                rgb.append((_spherical_harmonics(entry['sh_degree'],model.get_xyz[start:stop]-campos,coeff)+.5).clamp_min(0))
            rgb=torch.cat(rgb)
            scales=model.get_scaling
            if scales.shape[1]==2:scales=torch.cat([scales,torch.ones(len(scales),1)],dim=1)
            kwargs=dict(means=model.get_xyz.cuda(),quats=model.get_rotation.cuda(),scales=scales.cuda(),opacities=model.get_opacity[:,0].cuda(),colors=rgb.cuda(),viewmats=pose[None].cuda(),Ks=K[None].cuda(),width=width,height=height,packed=False,backgrounds=torch.zeros(1,3,device='cuda'),near_plane=.01,far_plane=10000.,render_mode='RGB')
            if entry['gaussian_type']=='2dgs':
                # This installed 2DGS version expects unpacked per-point RGB.
                kwargs['packed']=False
                kwargs['colors']=kwargs['colors'][None]
                result=rasterization_2dgs(**kwargs)
            else:
                result=rasterization(**kwargs,rasterize_mode='antialiased')
            arr=result[0][0].clamp(0,1).cpu().numpy()
            Image.fromarray(np.uint8(np.round(arr*255))).save(dest)
            entry['render']={'version':2,'width':width,'height':height,'primitives':len(model.get_xyz),'renderer':'Native gsplat 2DGS or 3DGS; CPU SH evaluation at the identical GT pose; black background; same clamp and resolution for every prior.','packed':kwargs['packed'],'pose':'same as map projection camera and real query GT','all_primitives':True,'sha256':digest(dest)}
        del kwargs,result,model,rgb,scales
        gc.collect();torch.cuda.empty_cache()
        PROV.write_text(json.dumps(p,indent=2))


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    p=read(PROV)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.1,4.5),facecolor='white')
    left=.018; gap=.013; cell=(.964-4*gap)/5
    query=Image.open(ASSETS/'real_query.png')
    neutral=np.load(ASSETS/'neutral_geometry.npz')
    K=np.array(p['camera']['K']); w=p['camera']['width']; h=p['camera']['height']
    x0,y0,x1,y1=p['neutral_geometry']['roi']
    ybottom=[.715,.472,.215]; heights=[cell*7.1/4.5*9/16]*3
    row_text=[('a','Input Gaussian prior'),('b','Compact landmarks · shared descriptor PCA'),('c','Localization · same real-query crop')]
    for j,(letter,title) in enumerate(row_text):
        fig.text(left,ybottom[j]+heights[j]+.026,f'({letter})  {title}',fontsize=7.3,weight='bold',color='#263344')
    for i,key in enumerate(KEYS):
        entry=p['priors'][key]; x=left+i*(cell+gap)
        # Column title stays with Gaussian image.
        fig.text(x+cell/2,.974,entry['label'],ha='center',va='top',fontsize=8.2,weight='bold')
        ax=fig.add_axes([x,ybottom[0],cell,heights[0]])
        ax.imshow(Image.open(ASSETS/(key+'_prior.png')));ax.axis('off')
        ax=fig.add_axes([x,ybottom[1],cell,heights[1]])
        d=np.load(ASSETS/(key+'_map.npz'))
        order=np.argsort(d['depth'])[::-1]
        ax.set_facecolor('#f5f7fa')
        ax.scatter(d['uv'][order,0],d['uv'][order,1],c=d['colors'][order],s=.105,lw=0,rasterized=True)
        ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        fig.text(x+cell/2,ybottom[1]-.014,f"N = {entry['landmarks']/1000:.1f}k",ha='center',va='top',fontsize=7,color='#45556b')
        ax=fig.add_axes([x,ybottom[2],cell,heights[2]])
        ax.imshow(query);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal');ax.axis('off')
        gt=neutral['gt_uv']; est,_=project(neutral['xyz'],entry['result']['pose_w2c'],K)
        inside=(gt[:,0]>=x0)&(gt[:,0]<x1)&(gt[:,1]>=y0)&(gt[:,1]<y1)
        # GT circles and estimate crosses show exact locations, without exaggeration.
        for start,end in zip(gt[inside],est[inside]):
            ax.plot([start[0],end[0]],[start[1],end[1]],color='white',lw=.55,alpha=.85)
        ax.scatter(gt[inside,0],gt[inside,1],s=10,facecolors='none',edgecolors=GT_COLOR,linewidths=.65)
        ax.scatter(est[inside,0],est[inside,1],s=8,marker='x',c=EST_COLOR,linewidths=.70)
        r=entry['result']
        fig.text(x+cell/2,ybottom[2]-.018,f"{r['translation_error_cm']:.2f} cm / {r['rotation_error_deg']:.3f}°",ha='center',va='top',fontsize=7,weight='bold')
    # Compact, column-aligned aggregate table avoids a fourth crowded plot.
    fig.text(left,.133,'(d)  Five-scene aggregate',weight='bold',fontsize=7.3,color='#263344')
    fig.text(.69,.403,'GT  ○',color=GT_COLOR,fontsize=7.1)
    fig.text(.805,.403,'Estimate  ×',color=EST_COLOR,fontsize=7.1)
    for i,key in enumerate(KEYS):
        x=left+i*(cell+gap); a=p['priors'][key]['aggregate']
        fig.text(x+cell/2,.090,f"TE  {a['TE_cm']:.2f} cm",ha='center',fontsize=7.0)
        fig.text(x+cell/2,.055,f"R$_{{5,5}}$  {a['R5_pct']:.1f}%",ha='center',fontsize=7.0)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/('E4_gaussian_prior.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
    caption=('Gaussian priors, compact landmark representations, and localization. Columns compare MAtCHA, vanilla 2DGS, vanilla 3DGS, i3DGS raw, and i3DGS after 1k refinement steps. '
             '(a) Original Gaussian priors rendered offline at one identical calibrated GT camera, resolution, and black background. '
             '(b) Evaluated compact landmarks projected into that camera; color is a single three-component PCA fit to 8,192 frozen SuperPoint descriptors per prior, with shared pooled 1–99 percentile RGB scaling. Colors are not semantic labels. N is the full scene map size. '
             '(c) An identical real-query crop overlays exactly the same neutral COLMAP SfM points: GT projection in green circles and each recorded pose projection in magenta crosses; white connectors show the exact displacement between the corresponding projections; labels report the recorded TE/RE. GT-only RGB texture, projected density, and a spatial grid determine display points; every prior projects identical point IDs. The crop enlarges the image equally across methods without exaggerating pose displacement. These are point projections, not surface edges or verified mesh-visible geometry. '
             'The query is selected from all 530 queries with valid poses under every prior: e=max(TE/5 cm, RE/5 degrees), within-prior percentile ranks are averaged, and the query nearest average rank 0.5 is selected (0.500189). The scene is fixed to the structure-rich StMarysChurch before query ranking. '
             '(d) Unweighted five-scene means of scene-median TE and scene recall R5,5 (5 cm, 5 degrees). MAtCHA uses the factorial v1_f1_c1_compact operating point consistently with the other main figures; the other producers use their archived prior-suite compact maps. Upstream preparation differs across priors; this is a compatibility and sensitivity comparison, not invariance to prior quality. Rendering is only for this figure and is absent from AnyGSLoc online localization.')
    (OUT/'E4_gaussian_prior.caption.txt').write_text(caption+'\n')
    p['outputs']={ext:{'path':str(OUT/('E4_gaussian_prior.'+ext)),'sha256':digest(OUT/('E4_gaussian_prior.'+ext))} for ext in ['pdf','svg','png']}
    if (ASSETS/'render_validation.json').exists():p['render_validation']=read(ASSETS/'render_validation.json')
    p['builder']={'path':str(Path(__file__).resolve()),'sha256':digest(__file__),
                  'phases':['prepare','render','validate','draw'],
                  'cpu_python':'/root/miniconda3/envs/i3dgs/bin/python',
                  'render_python':'/root/miniconda3/envs/g4splat/bin/python',
                  'render_environment':{'PATH_prefix':'/root/miniconda3/envs/g4splat/bin','LD_LIBRARY_PATH':'/root/miniconda3/envs/g4splat/lib:/usr/local/cuda/lib64','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
                  'source_mutation':False,'new_algorithm_experiments':False}
    PROV.write_text(json.dumps(p,indent=2))


def validate():
    """Compare memory-bounded rendering to the frozen native renderer."""
    import torch
    from PIL import Image
    from anygsloc.priors.models import GaussianModel2D
    from anygsloc.priors.rendering import render_from_pose_gsplat
    torch.set_num_threads(4)
    d=read(PROV);e=d['priors']['matcha'];c=d['camera'];k=np.array(c['K'])
    m=GaussianModel2D(3);m.load_ply(e['ply'],loc_feature_dim=0)
    with torch.inference_mode():
        v=render_from_pose_gsplat(m,torch.tensor(c['gt_pose_w2c'],dtype=torch.float32,device='cuda'),2*np.arctan(c['width']/(2*k[0,0])),2*np.arctan(c['height']/(2*k[1,1])),640,360,bg_color=torch.zeros(3,device='cuda'),render_mode='RGB',rgb_only=True)
        rgb=np.uint8(np.round(v['render'][:3].clamp(0,1).cpu().numpy().transpose(1,2,0)*255))
    ref=np.asarray(Image.open(ASSETS/'matcha_prior.png'))
    diff=np.abs(rgb.astype(float)-ref.astype(float))
    native_path=ASSETS/'matcha_native_reference.png'
    Image.fromarray(rgb).save(native_path)
    check={'MAtCHA_native_renderer_max_uint8_difference':float(diff.max()),'mean_uint8_difference':float(diff.mean()),'fraction_equal':float((diff==0).mean()),'native_reference':str(native_path),'native_reference_sha256':digest(native_path),'comparison':'Native SH-on-GPU renderer vs memory-bounded CPU-SH figure renderer, all original primitives, identical camera/resolution.'}
    assert diff.mean()<.001 and diff.max()<=8,check
    (ASSETS/'render_validation.json').write_text(json.dumps(check,indent=2))
    print(check,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','render','validate','draw']);args=parser.parse_args()
    globals()[args.phase]()
