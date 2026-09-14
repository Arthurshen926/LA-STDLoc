"""Matched-camera MAtCHA rendering beside an explicitly historical G4Splat run."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from common.hashing import sha256_file
from data.datasets import ColmapDataset
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_from_pose_gsplat


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior',type=Path);p.add_argument('--gaussian-type',choices=['2dgs','3dgs'],default='2dgs')
    p.add_argument('--label',default='MAtCHA');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35,0)
    historical=Path('/mnt/pool/sqy/G4Splat_runs/cambridge_moge3_static_canopy_v1/validation_v113_mixed_foliage24_2k')
    registry=json.loads((historical/'cameras.json').read_text())
    prior=Path('/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/StMarysChurch/prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply')
    if a.prior is not None:prior=a.prior
    dataset=ColmapDataset('/mnt/pool/sqy/Cambridge_stdloc/StMarysChurch',images='processed')
    cameras={c.image_name:c for c in dataset.split('all')}
    model=(GaussianModel2D if a.gaussian_type=='2dgs' else GaussianModel3D)(3,device=torch.device('cuda:0'));model.load_ply(prior,loc_feature_dim=0);model.eval()
    rows=[];panels=[]
    for index in [632,633,634,408]:
        camera=registry[index];name=camera['img_name'].replace('__','/')+'.png';c=cameras[name]
        rotation=np.array(camera['rotation']);position=np.array(camera['position'])
        w2c=np.eye(4);w2c[:3,:3]=rotation.T;w2c[:3,3]=-rotation.T@position
        pose_delta=float(np.max(np.abs(w2c-c.pose_w2c)))
        # Dataset pose is float32, historical JSON is double precision.
        if not np.allclose(w2c,c.pose_w2c,atol=5e-6,rtol=0):raise ValueError('camera frame mismatch')
        w,h=camera['width'],camera['height'];fx,fy=camera['fx'],camera['fy']
        if not np.allclose([fx,fy],[c.width/(2*math.tan(c.fov_x/2))*w/c.width,c.height/(2*math.tan(c.fov_y/2))*h/c.height],atol=1e-5):
            raise ValueError('camera focal mismatch')
        package=render_from_pose_gsplat(model,torch.tensor(w2c,dtype=torch.float32,device='cuda:0'),
            2*math.atan(w/(2*fx)),2*math.atan(h/(2*fy)),w,h,bg_color=torch.zeros(3,device='cuda:0'),
            render_mode='RGB+ED',rgb_only=True,rasterize_mode='antialiased')
        rgb=(package['render'][:3].clamp(0,1)*255).round().byte().permute(1,2,0).cpu().numpy()
        render=Image.fromarray(rgb)
        if a.label=='MAtCHA':render.save(a.output/f'{index:05d}_matcha.png')
        render.save(a.output/f'{index:05d}_render.png')
        refpath=historical/f'visualization/{index:05d}_gt.png';gpath=historical/f'visualization/{index:05d}_canonical.png'
        ref=Image.open(refpath).convert('RGB');g=Image.open(gpath).convert('RGB')
        glabel='G4Splat v113 mixed foliage24 2k (historical)'
        recent=historical.parent/f'diagnostic_v310_source_sh_slow800/view_{index}_0800.png'
        if recent.exists():
            pair=Image.open(recent).convert('RGB')
            if pair.size!=(2*w,h):raise ValueError('v310 pair layout mismatch')
            ref=pair.crop((0,0,w,h));g=pair.crop((w,0,2*w,h))
            refpath=recent;gpath=recent;glabel='G4Splat v310 source SH slow / step800'
        if ref.size!=(w,h) or g.size!=(w,h):raise ValueError('historical image extent mismatch')
        # Fixed common crop for the canonical seq2 views; no per-method selection.
        crop=(int(.28*w),int(.12*h),int(.90*w),int(.88*h))
        panel=Image.new('RGB',(w*3,h+32));draw=ImageDraw.Draw(panel)
        for col,(im,label) in enumerate([(ref,'Reference RGB'),(render,a.label+' / gsplat / black'),(g,glabel)]):
            panel.paste(im,(col*w,32));draw.text((col*w+5,8),label,fill='white')
        panel.save(a.output/f'{index:05d}_comparison.png');panels.append(panel)
        zoom=Image.new('RGB',(w*3,h+32));draw=ImageDraw.Draw(zoom)
        for col,(im,label) in enumerate([(ref,'Reference / common crop'),(render,a.label+' / common crop'),(g,'Historical G4Splat / common crop')]):
            zoom.paste(im.crop(crop).resize((w,h)),(col*w,32));draw.text((col*w+5,8),label,fill='white')
        zoom.save(a.output/f'{index:05d}_crop.png')
        rows.append({'index':index,'image':name,'pose_absolute_tolerance':5e-6,'pose_max_abs_delta':pose_delta,'intrinsics_equal':True,'crop':crop,'g4_label':glabel,
                     'reference':str(refpath),'reference_sha256':sha256_file(refpath),
                     'historical_g4':str(gpath),'historical_g4_sha256':sha256_file(gpath)})
    grid=Image.new('RGB',(panels[0].width,sum(x.height for x in panels)))
    y=0
    for panel in panels:grid.paste(panel,(0,y));y+=panel.height
    grid.save(a.output/'comparison_grid.png')
    report={'prior':str(prior),'prior_sha256':sha256_file(prior),'rows':rows,'label':a.label,'gaussian_type':a.gaussian_type,
            'purpose':'matched-view canopy/occlusion qualitative diagnostic, not method ranking',
            'g4_is_latest':False,'renderer':'gsplat antialiased, black background',
            'caveats':['Historical G4 render uses its original renderer/settings; backgrounds are not harmonized.',
                       'Views may be reconstruction training views; not novel-view quality or localization evaluation.',
                       'No unseen background ground truth can be inferred behind opaque canopy.']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
