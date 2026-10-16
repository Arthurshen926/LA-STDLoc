"""Third-round CVPR prior figure; frozen outputs + offline visualization only."""
from pathlib import Path
import argparse, gc, hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from build_cvpr_e4_prior import digest, read, project, KEYS, LABELS
POOL=Path('/mnt/pool/sqy');OUT=ROOT/'paper/experiments_cvpr_r3_20261003';A=OUT/'assets/e4'
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
    d['selected']=['StMarysChurch_02','ShopFacade_04']
    d['selection']['final']='Two cases from the frozen R2 28-query shortlist, maintaining all four non-raw producers below 5 cm and 0.3 deg; among church/shop views, select visible raw pose misalignment. StMarysChurch seq13/frame00161 and ShopFacade seq3/frame00047. The Church/Shop pair also retains distinct appearance behavior. Selection is illustrative and conditional, not representative.'
    d['selection']['visual_observation']='The raw poses cause clear seam discontinuities in a common reference model, while the four non-raw priors align closely. The prior-render row retains actual appearance differences without making a general rendering-quality ranking.'
    d['selection']['r3_replacement_reason']='R2 ShopFacade raw TE=4.78cm and RE=0.078deg was already well aligned; centimeter-scale differences at distant structures were also suppressed by the 480x270-pixel crop and white seams. R3 uses genuinely misaligned raw outputs and one equal 240x150-pixel crop per case, without altering any camera pose.'
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
            target=A/(cid+'_'+key+'_reference_full_2880.png')
            if target.exists():continue
            print('Reference pose',cid,key,flush=True)
            pose=item['gt_pose_w2c'] if key=='gt' else item['results'][key]['pose_w2c']
            Image.fromarray(render_model(model,item,pose,width=2880)).save(target)
        del model;gc.collect();torch.cuda.empty_cache()
        gt=np.asarray(Image.open(A/(cid+'_gt_reference_full_2880.png')))
        # The same tighter crop is applied to all five estimates. Candidate ROIs
        # are ranked by raw-vs-GT RGB difference times capped GT edge strength;
        # a structural ROI among the top three non-overlapping proposals is
        # chosen visually. This is disclosed illustrative selection, not a metric.
        scale=2880/item['width'];cw,ch=360,225
        x,y={'StMarysChurch_02':(1629,765),'ShopFacade_04':(285,861)}[cid]
        crop=[x,y,x+cw,y+ch]
        item['alignment']={'reference_prior':'MAtCHA, identical frozen model for GT and all five estimated poses','reference_ply':d['scenes'][scene]['priors']['matcha']['ply'],'full_render_width':2880,'crop_render_px':crop,'crop_native_px':[round(v/scale) for v in crop],'crop_rule':'At 1920-wide reference renders, from a central fixed grid rank 240x150 ROI by mean absolute raw-vs-GT RGB difference times capped GT Sobel-gradient magnitude; exclude >1% near-black pixels. Select an architectural contour among the top three non-overlapping proposals after visual inspection; all priors use that identical crop. Final common reference renders are supersampled to 2880 wide, with the exact same camera/view/crop field of view (360x225 pixels).','composition':'Four alternating horizontal bands: GT / estimated / GT / estimated; no seam line is drawn. Exact camera poses, no displacement scaling.','renderer':'All original Gaussians, original SH color, same calibration, black background and output resolution; CPU SH is the memory-bounded equivalent of the native rasterizer.'}
        Image.fromarray(gt[y:y+ch,x:x+cw]).save(A/(cid+'_gt_reference_crop.png'))
        for key in KEYS:
            estimated=np.asarray(Image.open(A/(cid+'_'+key+'_reference_full_2880.png')))[y:y+ch,x:x+cw]
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
    d=read(P);fw,fh=7.1,6.3;fig=plt.figure(figsize=(fw,fh),facecolor='white')
    margin=.1;qwidth=.92;gap=.065;pw=(fw-2*margin-qwidth-5*gap)/5
    xs=[margin]+[margin+qwidth+gap+i*(pw+gap) for i in range(5)];widths=[qwidth]+[pw]*5
    def text_in(x,y,text,**kw):return fig.text(x/fw,1-y/fh,text,**kw)
    def panel(x,top,width,height):return fig.add_axes([x/fw,1-(top+height)/fh,width/fw,height/fh])
    for j,label in enumerate(['Real query']+LABELS):
        text_in(xs[j]+widths[j]/2,.075,label,ha='center',va='top',weight='bold',fontsize=7.8,color=INK)
    for ci,cid in enumerate(d['selected']):
        item=next(x for x in d['candidates'] if x['id']==cid);scene=item['scene'];offset=ci*3.1
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
        gt=Image.open(A/(cid+'_gt_reference_crop.png'));ah=pw*150/240
        ax=panel(xs[0],tops[2]+(ah-qwidth*150/240)/2,qwidth,qwidth*150/240);ax.imshow(gt);ax.axis('off')
        ax.plot([0,0],[0,gt.height],color=GREEN,lw=2,clip_on=False)
        text_in(xs[0]+qwidth/2,tops[2]+ah+.095,'GT reference',ha='center',va='top',color=GREEN,fontsize=7)
        for j,key in enumerate(KEYS,1):
            src=d['scenes'][scene]['priors'][key];x=xs[j]
            ax=panel(x,tops[0],pw,ih);ax.imshow(Image.open(A/(cid+'_'+key+'_prior.png')));ax.axis('off')
            ax=panel(x,tops[1],pw,ih);m=np.load(A/(cid+'_'+key+'_map.npz'));order=np.argsort(m['depth'])[::-1]
            ax.set_facecolor('#f6f7f9');ax.scatter(m['uv'][order,0],m['uv'][order,1],c=m['colors'][order],s=.08,lw=0,rasterized=True)
            ax.set_xlim(0,item['width']);ax.set_ylim(item['height'],0);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            text_in(x+pw/2,tops[1]+ih+.075,f"N = {src['landmark_count']/1000:.1f}k",ha='center',va='top',color='#607083',fontsize=7)
            im=Image.open(A/(cid+'_'+key+'_alignment.png'));ax=panel(x,tops[2],pw,ah);ax.imshow(im,interpolation='nearest');ax.axis('off')
            cuts=np.linspace(0,im.height,5)
            for band in range(4):
                ax.plot([0,0],[cuts[band],cuts[band+1]],color=GREEN if band%2==0 else MAGENTA,lw=2,clip_on=False)
            r=item['results'][key]
            text_in(x+pw/2,tops[2]+ah+.095,f"{r['translation_error_cm']:.2f} cm / {r['rotation_error_deg']:.3f}°",ha='center',va='top',fontsize=7,color=INK)
        if ci==0:
            fig.add_artist(plt.Line2D([margin/fw,(fw-margin)/fw],[1-3.22/fh]*2,transform=fig.transFigure,color='#dbe0e6',lw=.6))
    for ext in ['pdf','svg','png']:fig.savefig(OUT/('E4_gaussian_prior.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
    caption=("Gaussian-prior compatibility in two illustrative cases. Each case shows prior RGB at the calibrated GT camera (top), compact landmark maps with one shared descriptor PCA and normalization (middle), and pose alignment in a common MAtCHA reference model (bottom). Four horizontal bands alternate GT (green edge) and the archived estimated pose (magenta edge); broken architectural contours expose actual pose disagreement. White rectangles locate the identical zoom used for every prior in that case. Four non-raw producers yield closely aligned estimates, whereas raw i3DGS produces visible discontinuities. Labels report the recorded translation/rotation errors; N denotes scene map size. "
             "The examples, StMarysChurch seq13/frame00161 and ShopFacade seq3/frame00047, are deliberately selected from the archived shortlist so that MAtCHA, 2DGS, 3DGS, and i3DGS+1k each satisfy TE<5 cm and RE<0.3 degrees, while raw disagreement is visually interpretable. They are not representative error samples. Each common zoom covers 240x150 original-camera pixels and is selected for structural visibility; final reference images are supersampled equally for print quality. No pose displacement is magnified or camera alignment altered. PCA is fitted once to equal samples from all ten scene/prior maps; colors encode descriptor directions, not semantics. MAtCHA uses the factorial v1_f1_c1_compact map and other producers use their archived prior-suite compact maps. Gaussian rendering is only an offline visualization tool and is absent from AnyGSLoc online localization. No training, reconstruction, or localization was rerun.")
    (OUT/'E4_gaussian_prior.caption.txt').write_text(caption+'\n')
    d['outputs']={e:{'path':str(OUT/('E4_gaussian_prior.'+e)),'sha256':digest(OUT/('E4_gaussian_prior.'+e))} for e in ['pdf','svg','png']}
    d['builder']={'path':str(Path(__file__).resolve()),'sha256':digest(__file__),'phases':['candidates','select','render_final','draw','validate'],
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
        'appearance_resolution':'640×360 for every prior','reference_resolution':'2880×1620 for GT and every estimated pose',
        'shared_alignment_crop':'360×225 pixels at render width 2880, equivalent to the same 240×150-pixel field at width 1920; same bounds and bands for all five priors'}
    d['limitations']=['Conditional favorable selection is not representative of all queries.','Query and crop selection prioritize visible raw misalignment among cases where the other four priors give similar accurate poses.','Upstream preparation conditions differ across prior producers.','Strip images compare a common reference model, not each localization prior appearance.','Exact centimeter-scale differences among the four non-raw estimates are expected to remain subtle at print scale.']
    P.write_text(json.dumps(d,indent=2))


def validate():
    from PIL import Image
    from anygsloc.evaluation.metrics import pose_error
    sys.path.insert(0,'/root/.cache/anygsloc_figure_deps')
    import fitz
    d=read(P);report={'new_localization_runs':0,'new_training_runs':0,'cases':{},'figure':{}}
    for cid in d['selected']:
        item=next(x for x in d['candidates'] if x['id']==cid);crop=item['alignment']['crop_render_px'];x,y,x1,y1=crop
        gt=np.asarray(Image.open(A/(cid+'_gt_reference_full_2880.png')))[y:y1,x:x1]
        assert np.array_equal(gt,np.asarray(Image.open(A/(cid+'_gt_reference_crop.png'))))
        stats={}
        for key in KEYS:
            src=d['scenes'][item['scene']]['priors'][key]
            assert digest(src['map'])==src['map_sha256']
            assert digest(src['results'])==src['results_sha256']
            pose=np.asarray(item['results'][key]['pose_w2c']);truth=np.asarray(item['gt_pose_w2c'])
            re,te=pose_error(pose,truth)
            assert abs(te-item['results'][key]['translation_error_cm'])<.002
            assert abs(re-item['results'][key]['rotation_error_deg'])<.0002
            pred=np.asarray(Image.open(A/(cid+'_'+key+'_reference_full_2880.png')))[y:y1,x:x1]
            mix=gt.copy();cuts=np.linspace(0,len(gt),5,dtype=int)
            for b in [1,3]:mix[cuts[b]:cuts[b+1]]=pred[cuts[b]:cuts[b+1]]
            assert np.array_equal(mix,np.asarray(Image.open(A/(cid+'_'+key+'_alignment.png'))))
            stats[key]={'recorded_TE_RE_verified':True,'map_and_result_hashes_verified':True,'exact_unwarped_band_composition_verified':True,'TE_cm':float(te),'RE_deg':float(re),'visualization_only_crop_mean_absolute_RGB_difference':float(np.abs(gt.astype(float)-pred).mean()/255)}
        report['cases'][cid]=stats
    doc=fitz.open(OUT/'E4_gaussian_prior.pdf');p=doc[0]
    spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
    assert len(doc)==1 and min(s['size'] for s in spans)>=6.99
    assert all(p.rect.contains(fitz.Rect(s['bbox'])) for s in spans)
    fonts=p.get_fonts(full=True);assert all(f[2]!='Type3' for f in fonts)
    assert all(len(doc.extract_font(f[0])[3])>0 for f in fonts)
    pw=(7.1-.2-.92-5*.065)/5
    report['figure']={'size_inches':[p.rect.width/72,p.rect.height/72],'PNG_dimensions':list(Image.open(OUT/'E4_gaussian_prior.png').size),'minimum_font_pt':min(s['size'] for s in spans),'all_text_inside_page':True,'fonts_embedded':True,'no_type3_fonts':True,'alignment_native_crop_PPI':360/pw,'prior_RGB_native_PPI':640/pw}
    report['files']={ext:digest(OUT/('E4_gaussian_prior.'+ext)) for ext in ['pdf','svg','png','caption.txt','provenance.json']}
    (OUT/'E4_gaussian_prior.validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['figure'],indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['candidates','shortlist','board','select','render_final','draw','validate']);args=a.parse_args();globals()[args.phase]()
