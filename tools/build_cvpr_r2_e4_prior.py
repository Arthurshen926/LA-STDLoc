"""Second-round CVPR prior figure; frozen outputs + offline visualization only."""
from pathlib import Path
import argparse, gc, hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from build_cvpr_e4_prior import digest, read, project, KEYS, LABELS
POOL=Path('/mnt/pool/sqy');OUT=ROOT/'paper/experiments_cvpr_r2_20261003';A=OUT/'assets/e4'
P=OUT/'E4_gaussian_prior.provenance.json'
SCENES=['StMarysChurch','OldHospital','GreatCourt','ShopFacade','KingsCollege']
INK='#243349';GREEN='#219653';MAGENTA='#cc2f83'


def candidates():
    import torch
    from PIL import Image
    from anygsloc.data.datasets import ColmapDataset
    torch.set_num_threads(4);A.mkdir(parents=True,exist_ok=True)
    suite=POOL/'anygsloc_cambridge_suite_v2_20260914';factorial=POOL/'anygsloc_observation_factorial_20260914'
    reg=read(factorial/'registry.json');jobs=read(suite/'gpu0/state.json')['jobs']
    old=read(ROOT/'paper/figures_v5/selection.json')
    d={'scope':'Frozen results and original maps; visualization only, no new localization/training.',
       'selection':{'intent':'Illustrative favorable localization cases, explicitly not representative error samples.',
                    'eligibility':'All five finite poses; MAtCHA, 2DGS, 3DGS, i3DGS+1k each TE<5cm and RE<0.3deg; raw is unrestricted.',
                    'shortlist':'Five equally spaced lexical indices among eligible queries per scene, plus greatest raw TE if absent; fixed before rendering.',
                    'appearance_scoring':'Low-resolution offline RGB-vs-query MAE and luminance SSIM are shortlist proxies only, not additional localization results or a benchmark of prior quality.'},
       'scenes':{},'candidates':[],'selected':[]}
    for scene in SCENES:
        records={};d['scenes'][scene]={'priors':{}}
        for key,label in zip(KEYS,LABELS):
            if key=='matcha':
                entry=next(r for r in reg if r['id']==scene+'_v1_f1_c1_compact');run=Path(entry['result']);mp=Path(entry['map'])
                v=next(v for v in old['views'] if v['scene']==scene)
                source={'ply':v['prior'],'ply_sha256':v['prior_sha256'],'gaussian_type':'2dgs','sh_degree':3}
            else:
                job=jobs[f'prior_{scene}_{key}'];cmd=job['command'];mp=Path(cmd[cmd.index('--map')+1]);run=suite/'runs'/f'prior_{scene}_{key}'
                # Original cache audit binds the rendered Gaussian source.
                audits=list((POOL/f'anygsloc_three_gpu_20260910/runs/{scene}_{key}/v2_audit').glob('*.pt'))
                if not audits:
                    deploy=torch.load(mp,map_location='cpu',weights_only=False,mmap=True)
                    original=torch.load(deploy['provenance']['source'],map_location='cpu',weights_only=False,mmap=True)
                    audits=[Path(original['provenance']['v8_v2_audit_shards'][0])]
                    del deploy,original
                audit=audits[0]
                b=torch.load(audit,map_location='cpu',weights_only=False,mmap=True)['input']
                source={'ply':b['gaussian_ply'],'ply_sha256':b['gaussian_ply_sha256'],'gaussian_type':b['gaussian_type'],'sh_degree':b['sh_degree'],'audit':str(audit)}
            manifest=read(run/'manifest.json');assert digest(mp)==manifest['map_sha256']
            source.update(label=label,map=str(mp),map_sha256=manifest['map_sha256'],results=str(run/'results.json'),results_sha256=digest(run/'results.json'))
            d['scenes'][scene]['priors'][key]=source
            records[key]={r['image_name']:r for r in read(run/'results.json') if r['success'] and r.get('pose_w2c') is not None and np.isfinite(r['translation_error_cm']) and np.isfinite(r['rotation_error_deg'])}
        names=sorted(set.intersection(*[set(r) for r in records.values()]))
        good=[n for n in names if all(records[k][n]['translation_error_cm']<5 and records[k][n]['rotation_error_deg']<.3 for k in KEYS if k!='i3dgs_raw')]
        d['scenes'][scene]['common_valid_queries']=len(names);d['scenes'][scene]['eligible_queries']=len(good)
        selected=sorted(set(good[round(t*(len(good)-1))] for t in np.linspace(0,1,5))|{max(good,key=lambda n:records['i3dgs_raw'][n]['translation_error_cm'])})
        ds=ColmapDataset(POOL/'Cambridge_stdloc'/scene,load_masks=False)
        for number,name in enumerate(selected):
            cam=ds.camera(name);stem=f'{scene}_{number:02d}';K=np.array([[cam.width/(2*np.tan(cam.fov_x/2)),0,cam.width/2],[0,cam.height/(2*np.tan(cam.fov_y/2)),cam.height/2],[0,0,1.]])
            item={'id':stem,'scene':scene,'query':name,'width':cam.width,'height':cam.height,'K':K.tolist(),'gt_pose_w2c':cam.pose_w2c.tolist(),'query_path':str(cam.image_path),'results':{k:records[k][name] for k in KEYS}}
            Image.open(cam.image_path).convert('RGB').resize((480,270)).save(A/(stem+'_query.png'))
            d['candidates'].append(item)
        print(scene,len(good),'eligible',len(selected),'shortlisted',flush=True)
    P.write_text(json.dumps(d,indent=2))


def load_model(source):
    from anygsloc.priors.models import FrozenGaussianModel
    assert digest(source['ply'])==source['ply_sha256']
    m=FrozenGaussianModel(source['sh_degree'],device='cpu');m.load_ply(source['ply'],loc_feature_dim=0)
    return m


def render_model(model,item,pose,width=480):
    import torch
    from gsplat import rasterization,rasterization_2dgs
    from gsplat.cuda._torch_impl import _spherical_harmonics
    torch.set_num_threads(4)
    height=round(width*item['height']/item['width']);kk=np.array(item['K'],dtype=np.float64);kk[:2]*=width/item['width']
    K=torch.tensor(kk,dtype=torch.float32);T=torch.tensor(pose,dtype=torch.float32);center=torch.linalg.inv(T)[:3,3]
    with torch.inference_mode():
        rgb=[]
        for start in range(0,len(model.get_xyz),65536):
            end=start+65536;coeff=torch.cat([model._features_dc[start:end],model._features_rest[start:end]],1)
            rgb.append((_spherical_harmonics(model.active_sh_degree,model.get_xyz[start:end]-center,coeff)+.5).clamp_min(0))
        rgb=torch.cat(rgb);scales=model.get_scaling;two=scales.shape[1]==2
        if two:scales=torch.cat([scales,torch.ones(len(scales),1)],1)
        kw=dict(means=model.get_xyz.cuda(),quats=model.get_rotation.cuda(),scales=scales.cuda(),opacities=model.get_opacity[:,0].cuda(),colors=rgb.cuda(),viewmats=T[None].cuda(),Ks=K[None].cuda(),width=width,height=height,packed=False,backgrounds=torch.zeros(1,3,device='cuda'),near_plane=.01,far_plane=10000.,render_mode='RGB')
        if two:kw['colors']=kw['colors'][None];result=rasterization_2dgs(**kw)
        else:result=rasterization(**kw,rasterize_mode='antialiased')
        out=np.uint8(np.round(result[0][0].clamp(0,1).cpu().numpy()*255))
        del kw,result,rgb,scales
    torch.cuda.empty_cache()
    return out


def score(ref,pred):
    from scipy.ndimage import gaussian_filter
    r=ref.astype(float)/255;p=pred.astype(float)/255
    mask=(r.max(2)>.02);mask[:8]=False;mask[-8:]=False;mask[:,:8]=False;mask[:,-8:]=False
    x=r.mean(2);y=p.mean(2);mx=gaussian_filter(x,1.5);my=gaussian_filter(y,1.5)
    vx=gaussian_filter(x*x,1.5)-mx*mx;vy=gaussian_filter(y*y,1.5)-my*my;cov=gaussian_filter(x*y,1.5)-mx*my
    s=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return {'mean_absolute_rgb_error':float(np.abs(r-p)[mask].mean()),'luminance_ssim':float(s[mask].mean())}


def shortlist():
    import torch
    from PIL import Image
    torch.set_num_threads(4);d=read(P)
    # Compare two optimized producers with the refined feed-forward producer.
    for scene in SCENES:
        items=[x for x in d['candidates'] if x['scene']==scene]
        for key in ['matcha','vanilla_3dgs','i3dgs_1k']:
            needed=[x for x in items if not (A/(x['id']+'_'+key+'_short.png')).exists()]
            model=load_model(d['scenes'][scene]['priors'][key]) if needed else None
            for item in items:
                path=A/(item['id']+'_'+key+'_short.png')
                if not path.exists():
                    print('shortlist',scene,key,item['query'],flush=True)
                    Image.fromarray(render_model(model,item,item['gt_pose_w2c'])).save(path)
                item.setdefault('appearance_proxy',{})[key]=score(np.asarray(Image.open(A/(item['id']+'_query.png'))),np.asarray(Image.open(path)))
            del model;gc.collect();torch.cuda.empty_cache()
            P.write_text(json.dumps(d,indent=2))
    for item in d['candidates']:
        a=item['appearance_proxy'];item['ff_minus_optimized_ssim']=a['i3dgs_1k']['luminance_ssim']-(a['matcha']['luminance_ssim']+a['vanilla_3dgs']['luminance_ssim'])/2
    P.write_text(json.dumps(d,indent=2))
    board()


def board():
    from PIL import Image,ImageDraw,ImageFont
    d=read(P);items=sorted(d['candidates'],key=lambda x:x['ff_minus_optimized_ssim'],reverse=True)
    for start in range(0,len(items),10):
        chunk=items[start:start+10];im=Image.new('RGB',(1440,len(chunk)*245),'white');draw=ImageDraw.Draw(im)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        for row,item in enumerate(chunk):
            text=f"{item['id']} {item['query']}  FF-minus-optimized SSIM proxy {item['ff_minus_optimized_ssim']:+.3f}"
            draw.text((5,row*245+2),text,fill=INK,font=font)
            for col,(key,label) in enumerate([(None,'Query'),('matcha','MAtCHA'),('vanilla_3dgs','3DGS'),('i3dgs_1k','i3DGS+1k')]):
                path=A/(item['id']+('_query.png' if key is None else '_'+key+'_short.png'))
                draw.text((col*360+5,row*245+25),label,fill=INK,font=font)
                im.paste(Image.open(path).resize((360,202)),(col*360,row*245+43))
        im.save(A/f'shortlist_board_{start//10}.jpg',quality=93)
    print('Ranked candidate proxies:',[(x['id'],round(x['ff_minus_optimized_ssim'],3)) for x in items],flush=True)


def select():
    import torch
    from PIL import Image
    torch.set_num_threads(4);d=read(P)
    d['selected']=[min(d['candidates'],key=lambda x:x['ff_minus_optimized_ssim'])['id'],max(d['candidates'],key=lambda x:x['ff_minus_optimized_ssim'])['id']]
    d['selection']['final']='Minimum and maximum FF-minus-(MAtCHA,3DGS mean) luminance-SSIM proxy within the predefined 28-query favorable shortlist; visually inspected. Proxy values are not displayed or promoted as paper quantitative results.'
    d['selection']['visual_observation']='StMarysChurch: optimized priors visibly retain clearer architecture. ShopFacade: i3DGS+1k shows fewer doubled window structures than the displayed 3DGS rendering; this does not establish superiority over every optimized prior.'
    feats={};xyz={};samples={};rng=np.random.default_rng(2026)
    for cid in d['selected']:
        item=next(x for x in d['candidates'] if x['id']==cid)
        Image.open(item['query_path']).convert('RGB').save(A/(cid+'_real_query.png'))
        for key in KEYS:
            src=d['scenes'][item['scene']]['priors'][key]
            mp=torch.load(src['map'],map_location='cpu',weights_only=False,mmap=True)
            xyz[cid+'_'+key]=mp['anchor_xyz'].numpy().copy();feats[cid+'_'+key]=mp['anchor_features'].numpy().copy()
            src['landmark_count']=len(mp['anchor_xyz'])
            pose=np.asarray(item['results'][key]['pose_w2c']);gt=np.asarray(item['gt_pose_w2c'])
            te=np.linalg.norm(np.linalg.inv(pose)[:3,3]-np.linalg.inv(gt)[:3,3])*100
            assert abs(te-item['results'][key]['translation_error_cm'])<.002
    count=min(4096,min(len(x) for x in feats.values()))
    for name,f in feats.items():samples[name]=rng.choice(len(f),count,replace=False)
    shared=np.concatenate([feats[name][ids] for name,ids in samples.items()]).astype(np.float64)
    mean=shared.mean(0);v,w=np.linalg.eigh((shared-mean).T@(shared-mean)/(len(shared)-1));pc=w[:,-3:][:,::-1]
    pc*=np.sign(pc[np.argmax(abs(pc),axis=0),np.arange(3)])[None]
    low,high=np.percentile((shared-mean)@pc,[1,99],axis=0)
    np.savez_compressed(A/'shared_pca.npz',mean=mean,components=pc,low=low,high=high,**samples)
    d['pca']={'fit':'One pooled PCA over equal counts from each of all ten scene/prior compact maps','samples_per_scene_prior':count,'seed':2026,'normalization':'Shared pooled 1–99 percentile per PC, clipped to [0,1]','colors_are_semantics':False}
    for cid in d['selected']:
        item=next(x for x in d['candidates'] if x['id']==cid)
        for key in KEYS:
            name=cid+'_'+key;f=feats[name];rgb=np.clip(((f-mean)@pc-low)/(high-low),0,1).astype(np.float32)
            uv,z=project(xyz[name],item['gt_pose_w2c'],item['K'])
            keep=(z>0)&(uv[:,0]>=0)&(uv[:,0]<item['width'])&(uv[:,1]>=0)&(uv[:,1]<item['height'])
            np.savez_compressed(A/(name+'_map.npz'),uv=uv[keep],depth=z[keep],colors=rgb[keep])
    d['map_pose_versions']='MAtCHA uses factorial v1_f1_c1_compact; other priors use archived prior-suite compact maps. Every figure map SHA matches its recorded evaluation manifest.'
    P.write_text(json.dumps(d,indent=2));print('Selected',d['selected'],'PCA n',count,flush=True)


def render_final():
    import torch
    from PIL import Image
    from scipy.ndimage import sobel
    torch.set_num_threads(4);d=read(P)
    for cid in d['selected']:
        item=next(x for x in d['candidates'] if x['id']==cid);scene=item['scene']
        for key in KEYS:
            target=A/(cid+'_'+key+'_prior.png')
            if target.exists():continue
            print('Final prior',cid,key,flush=True);model=load_model(d['scenes'][scene]['priors'][key])
            Image.fromarray(render_model(model,item,item['gt_pose_w2c'],width=640)).save(target)
            del model;gc.collect();torch.cuda.empty_cache()
        model=load_model(d['scenes'][scene]['priors']['matcha'])
        for key in ['gt']+KEYS:
            target=A/(cid+'_'+key+'_reference_full.png')
            if target.exists():continue
            print('Reference pose',cid,key,flush=True)
            pose=item['gt_pose_w2c'] if key=='gt' else item['results'][key]['pose_w2c']
            Image.fromarray(render_model(model,item,pose,width=1920)).save(target)
        del model;gc.collect();torch.cuda.empty_cache()
        gt=np.asarray(Image.open(A/(cid+'_gt_reference_full.png')))
        # GT reference alone chooses the shared zoom: dense architectural texture,
        # no estimated pose, source-map density, or method-specific image enters.
        g=gt.astype(float).mean(2)/255;grad=np.abs(sobel(g,axis=1));grad=np.minimum(grad,np.percentile(grad,95))
        scale=1920/item['width'];cw,ch=round(480*scale),round(270*scale);choices=[]
        for y in range(round(gt.shape[0]*.08),round(gt.shape[0]*.92)-ch,24):
            for x in range(round(gt.shape[1]*.08),round(gt.shape[1]*.92)-cw,32):
                region=g[y:y+ch,x:x+cw]
                if (region<.025).mean()>.02:continue
                choices.append((float(grad[y:y+ch,x:x+cw].mean()),-abs(x+cw/2-gt.shape[1]/2),x,y))
        _,_,x,y=max(choices);crop=[x,y,x+cw,y+ch]
        item['alignment']={'reference_prior':'MAtCHA, identical frozen model for GT and all five estimated poses','reference_ply':d['scenes'][scene]['priors']['matcha']['ply'],'full_render_width':1920,'crop_render_px':crop,'crop_native_px':[round(v/scale) for v in crop],'crop_rule':'Maximum mean capped horizontal Sobel gradient of GT reference RGB on a fixed central grid; exclude >2% near-black pixels; no estimated pose used.','composition':'Four alternating horizontal bands: GT / estimated / GT / estimated. Exact camera poses, no displacement scaling.','renderer':'All original Gaussians, original SH color, same calibration, black background and output resolution; CPU SH is the memory-bounded equivalent of the native rasterizer.'}
        Image.fromarray(gt[y:y+ch,x:x+cw]).save(A/(cid+'_gt_reference_crop.png'))
        for key in KEYS:
            estimated=np.asarray(Image.open(A/(cid+'_'+key+'_reference_full.png')))[y:y+ch,x:x+cw]
            mix=gt[y:y+ch,x:x+cw].copy()
            cuts=np.linspace(0,ch,5,dtype=int)
            for band in [1,3]:mix[cuts[band]:cuts[band+1]]=estimated[cuts[band]:cuts[band+1]]
            Image.fromarray(mix).save(A/(cid+'_'+key+'_alignment.png'))
        P.write_text(json.dumps(d,indent=2))


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    d=read(P);fw,fh=7.1,6.12;fig=plt.figure(figsize=(fw,fh),facecolor='white')
    margin=.1;qwidth=.92;gap=.065;pw=(fw-2*margin-qwidth-5*gap)/5
    xs=[margin]+[margin+qwidth+gap+i*(pw+gap) for i in range(5)];widths=[qwidth]+[pw]*5
    def text_in(x,y,text,**kw):return fig.text(x/fw,1-y/fh,text,**kw)
    def panel(x,top,width,height):return fig.add_axes([x/fw,1-(top+height)/fh,width/fw,height/fh])
    for j,label in enumerate(['Real query']+LABELS):
        text_in(xs[j]+widths[j]/2,.075,label,ha='center',va='top',weight='bold',fontsize=7.8,color=INK)
    for ci,cid in enumerate(d['selected']):
        item=next(x for x in d['candidates'] if x['id']==cid);scene=item['scene'];offset=ci*2.95
        text_in(margin,.34+offset,f"({chr(97+ci)})  {scene}",fontsize=8,weight='bold',color=INK)
        # One shared legend; colored borders identify the horizontal source bands.
        text_in(4.94,.34+offset,'GT',color=GREEN,fontsize=7)
        text_in(5.34,.34+offset,'Estimated pose',color=MAGENTA,fontsize=7)
        tops=[.55+offset,1.31+offset,2.24+offset];ih=pw*9/16
        photo=Image.open(A/(cid+'_real_query.png'))
        ax=panel(xs[0],tops[0]+(ih-qwidth*9/16)/2,qwidth,qwidth*9/16);ax.imshow(photo);ax.axis('off')
        roi=item['alignment']['crop_native_px'];ax.add_patch(Rectangle((roi[0],roi[1]),roi[2]-roi[0],roi[3]-roi[1],fill=False,ec='white',lw=.7))
        text_in(xs[0]+qwidth/2,tops[1]+ih*.36,'Compact map',ha='center',va='center',weight='bold',color=INK)
        text_in(xs[0]+qwidth/2,tops[1]+ih*.60,'Shared PCA',ha='center',va='center',color='#607083')
        gt=Image.open(A/(cid+'_gt_reference_crop.png'))
        ax=panel(xs[0],tops[2]+(ih-qwidth*9/16)/2,qwidth,qwidth*9/16);ax.imshow(gt);ax.axis('off')
        ax.plot([0,0],[0,gt.height],color=GREEN,lw=2,clip_on=False)
        text_in(xs[0]+qwidth/2,tops[2]+ih+.095,'GT reference',ha='center',va='top',color=GREEN,fontsize=7)
        for j,key in enumerate(KEYS,1):
            src=d['scenes'][scene]['priors'][key];x=xs[j]
            ax=panel(x,tops[0],pw,ih);ax.imshow(Image.open(A/(cid+'_'+key+'_prior.png')));ax.axis('off')
            ax=panel(x,tops[1],pw,ih);m=np.load(A/(cid+'_'+key+'_map.npz'));order=np.argsort(m['depth'])[::-1]
            ax.set_facecolor('#f6f7f9');ax.scatter(m['uv'][order,0],m['uv'][order,1],c=m['colors'][order],s=.08,lw=0,rasterized=True)
            ax.set_xlim(0,item['width']);ax.set_ylim(item['height'],0);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            text_in(x+pw/2,tops[1]+ih+.075,f"N = {src['landmark_count']/1000:.1f}k",ha='center',va='top',color='#607083',fontsize=7)
            im=Image.open(A/(cid+'_'+key+'_alignment.png'));ax=panel(x,tops[2],pw,ih);ax.imshow(im);ax.axis('off')
            cuts=np.linspace(0,im.height,5)
            for band in range(4):
                ax.plot([0,0],[cuts[band],cuts[band+1]],color=GREEN if band%2==0 else MAGENTA,lw=2,clip_on=False)
            for yy in cuts[1:-1]:ax.axhline(yy,color='white',lw=.3,alpha=.7)
            r=item['results'][key]
            text_in(x+pw/2,tops[2]+ih+.095,f"{r['translation_error_cm']:.2f} cm / {r['rotation_error_deg']:.3f}°",ha='center',va='top',fontsize=7,color=INK)
        if ci==0:
            fig.add_artist(plt.Line2D([margin/fw,(fw-margin)/fw],[1-3.13/fh]*2,transform=fig.transFigure,color='#dbe0e6',lw=.6))
    for ext in ['pdf','svg','png']:fig.savefig(OUT/('E4_gaussian_prior.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
    caption=("Gaussian-prior compatibility in two illustrative localization-success cases. Columns compare MAtCHA, vanilla 2DGS, vanilla 3DGS, i3DGS raw, and i3DGS+1k. For each case, top: prior RGB rendered at the identical calibrated GT camera; middle: compact landmark maps colored by one shared descriptor PCA fitted once to equal samples from all ten scene/prior maps, with shared percentile normalization; bottom: pose alignment using one common MAtCHA reference Gaussian model per scene. Four horizontal bands alternate between the GT camera (green edge) and each archived estimated camera (magenta edge), so a discontinuity across a seam indicates pose-induced disagreement in the same reference appearance. The left image is the shared GT reference crop; white rectangles on the real queries identify the crop. No points are synthesized and no pose displacement is exaggerated. Labels report each recorded TE/RE; N denotes scene map size. "
             "These are deliberately selected success examples, not median-error samples: MAtCHA, 2DGS, 3DGS, and i3DGS+1k each satisfy TE<5 cm and RE<0.3 degrees, while raw i3DGS is unrestricted. Among 28 predefined, spatially varied lexical-shortlist queries across five scenes satisfying this condition, two views with opposite extremes of an offline appearance-similarity proxy were inspected and selected: StMarysChurch seq13/frame00117 and ShopFacade seq3/frame00043. The church view has visibly clearer architecture in the optimized priors; the shop view illustrates fewer doubled window structures for i3DGS+1k than for the displayed 3DGS rendering, without implying superiority over every optimized prior. Proxy scores are only recorded in the selection manifest and are not additional paper performance results. "
             "PCA colors encode frozen SuperPoint descriptor directions rather than semantics. MAtCHA results/maps use factorial v1_f1_c1_compact, and the other producers use archived prior-suite compact maps. Gaussian rendering is strictly an offline figure tool and is absent from AnyGSLoc online localization. No training, map reconstruction, or localization was rerun.")
    (OUT/'E4_gaussian_prior.caption.txt').write_text(caption+'\n')
    d['outputs']={e:{'path':str(OUT/('E4_gaussian_prior.'+e)),'sha256':digest(OUT/('E4_gaussian_prior.'+e))} for e in ['pdf','svg','png']}
    d['builder']={'path':str(Path(__file__).resolve()),'sha256':digest(__file__),'phases':['candidates','shortlist','select','render_final','draw'],
                  'CPU_python':'/root/miniconda3/envs/i3dgs/bin/python',
                  'render_python':'/root/miniconda3/envs/g4splat/bin/python',
                  'render_environment':{'PATH_prefix':'/root/miniconda3/envs/g4splat/bin','LD_LIBRARY_PATH':'/root/miniconda3/envs/g4splat/lib:/usr/local/cuda/lib64','CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}}
    d['asset_audit']={'shared_pca_sha256':digest(A/'shared_pca.npz'),'selected':{}}
    for cid in d['selected']:
        item=next(x for x in d['candidates'] if x['id']==cid)
        d['asset_audit']['selected'][cid]={'query_sha256':digest(item['query_path']),
            'assets':{path.name:digest(path) for path in sorted(A.glob(cid+'_*')) if path.is_file()}}
    d['visualization_constraints']={'all_Gaussians_rendered':True,'pose_displacement_exaggeration':False,
        'per_prior_color_or_contrast_adjustment':False,'new_localization_or_training':False,
        'appearance_resolution':'640×360 for every prior','reference_resolution':'1920×1080 for GT and every estimated pose',
        'shared_alignment_crop':'480×270 native pixels for each case; same bounds and bands for all five priors'}
    d['limitations']=['Conditional favorable selection is not representative of all queries.','Offline appearance proxy does not constitute a validated rendering-quality benchmark.','Upstream preparation conditions differ across prior producers.','Strip images compare a common reference model, not each localization prior appearance.','Exact centimeter-scale pose differences can remain subtle at print scale.']
    P.write_text(json.dumps(d,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['candidates','shortlist','board','select','render_final','draw']);args=a.parse_args();globals()[args.phase]()
