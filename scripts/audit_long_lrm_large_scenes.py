"""Mapping-only camera/export audit: no test poses or tuning from query GT."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from data.datasets import ColmapDataset
from priors.models import GaussianModel3D
from priors.rendering import render_from_pose_gsplat


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/anygsloc_three_gpu_20260910');reports=[]
    for scene in ['StMarysChurch','GreatCourt','ShopFacade']:
        inp=Path('/mnt/pool/sqy/anygsloc_corrected_cambridge_matrix_20260909/inputs')/scene/'long_lrm_input_pose32'
        manifest=json.loads((inp/'anygsloc_long_lrm_input.json').read_text())
        frames=json.loads((inp/'opencv_cameras.json').read_text())['frames']
        norm=manifest['long_lrm_internal_pose_normalization'];rigid=np.asarray(norm['rigid_world_to_normalized']);scale=norm['scene_scale']
        dataset=ColmapDataset(manifest['dataset_root'],images='processed');mapping=dataset.split('mapping');byname={c.image_name:c for c in mapping}
        selected=[r['source_name'] for r in manifest['records']]
        c2ws=np.linalg.inv(np.stack([f['w2c'] for f in frames]).astype(np.float32))
        # Independently reproduce upstream Torch normalization, not the adapter.
        t=torch.from_numpy(c2ws);forward=torch.nn.functional.normalize(t[:,:3,2].sum(0),dim=0)
        right=torch.nn.functional.normalize(torch.cross(t[:,:3,1].sum(0),forward,dim=0),dim=0)
        down=torch.nn.functional.normalize(torch.cross(forward,right,dim=0),dim=0)
        avg=torch.eye(4);avg[:3,:4]=torch.stack([right,down,forward,t[:,:3,3].mean(0)],dim=1)
        actual=torch.linalg.inv(avg)[None]@t;actual[:,:3,3]/=1.35*actual[:,:3,3].abs().max()
        centers=np.linalg.inv(np.stack([c.pose_w2c for c in mapping]))[:,:3,3]
        nearest=np.linalg.norm(centers[:,None]-c2ws[None,:,:3,3],axis=-1).min(1)
        camera_delta=max(float(np.abs(np.asarray(f['w2c'])-byname[n].pose_w2c).max()) for f,n in zip(frames,selected))
        native=GaussianModel3D(3);native.load_ply(root/f'priors/{scene}/long_lrm/native/gaussians.ply',loc_feature_dim=0)
        world=GaussianModel3D(3);world.load_ply(root/f'priors/{scene}/long_lrm/world.ply',loc_feature_dim=0)
        report={'scene':scene,'input_sizes':sorted(set((f['w'],f['h']) for f in frames)),
          'input_pose_max_abs_delta':camera_delta,'normalization_max_abs_delta':float(np.abs(actual.numpy()-np.asarray(norm['normalized_c2ws'])).max()),
          'scene_scale':scale,'mean_forward_norm':float(np.linalg.norm(c2ws[:,:3,2].mean(0))),
          'mapping_nearest_input_distance_m_p50_p90_max':np.quantile(nearest,[.5,.9,1]).tolist(),'views':[]}
        names=[selected[i] for i in [0,15,31]]+[mapping[i].image_name for i in np.argsort(nearest)[-3:]]
        for index,name in enumerate(names):
            c=byname[name];pose=np.asarray(c.pose_w2c);local_c2w=rigid@np.linalg.inv(pose);local_c2w[:3,3]*=scale
            h=round(c.height*640/c.width);ims=[]
            for model,view in [(native,np.linalg.inv(local_c2w)),(world,pose)]:
                pkg=render_from_pose_gsplat(model,torch.tensor(view,dtype=torch.float32,device='cuda'),c.fov_x,c.fov_y,640,h,
                    bg_color=torch.zeros(3,device='cuda'),render_mode='RGB',rgb_only=True,rasterize_mode='antialiased',
                    near_plane=.01*scale if model is native else .01,far_plane=10000*scale if model is native else 10000)
                ims.append(pkg['render'].clamp(0,1).permute(1,2,0).cpu().numpy());del pkg
            ref=np.asarray(Image.open(Path(manifest['dataset_root'])/'processed'/name).convert('RGB').resize((640,h)),dtype=np.float32)/255
            mse=float(np.mean((ims[0]-ims[1])**2));refmse=float(np.mean((ims[1]-ref)**2))
            row={'image':name,'is_input':name in selected,'canonical_world_psnr':-10*math.log10(max(mse,1e-15)),
                 'world_reference_psnr':-10*math.log10(max(refmse,1e-15)),'canonical_world_mae':float(np.abs(ims[0]-ims[1]).mean())}
            report['views'].append(row)
            panel=Image.new('RGB',(1920,h+28));draw=ImageDraw.Draw(panel)
            for col,(im,label) in enumerate([(ref,'mapping RGB'),(ims[0],'Long-LRM canonical'),(ims[1],'Long-LRM exported world')]):
                panel.paste(Image.fromarray(np.round(im*255).astype(np.uint8)),(640*col,28));draw.text((640*col+5,6),label,fill='white')
            panel.save(a.output/f'{scene}_{index}.png');print(scene,row,flush=True)
        reports.append(report);del native,world;torch.cuda.empty_cache()
        (a.output/'report.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
