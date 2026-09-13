"""Separate native Long-LRM rendering settings from reconstruction quality."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from data.datasets import ColmapDataset
from priors.models import GaussianModel3D
from priors.rendering import render_from_pose_gsplat


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.2)
    reports=[]
    for scene in ['StMarysChurch','GreatCourt','ShopFacade']:
        inp=Path('/mnt/pool/sqy/anygsloc_corrected_cambridge_matrix_20260909/inputs')/scene/'long_lrm_input_pose32'
        manifest=json.loads((inp/'anygsloc_long_lrm_input.json').read_text());scale=manifest['long_lrm_internal_pose_normalization']['scene_scale']
        dataset=ColmapDataset(manifest['dataset_root'],images='processed');cams={c.image_name:c for c in dataset.split('mapping')}
        model=GaussianModel3D(3);model.load_ply(Path('/mnt/pool/sqy/anygsloc_three_gpu_20260910')/f'priors/{scene}/long_lrm/world.ply',loc_feature_dim=0)
        rows=[]
        for index in [0,15,31]:
            name=manifest['records'][index]['source_name'];c=cams[name];h=round(c.height*640/c.width)
            ref=np.asarray(Image.open(Path(manifest['dataset_root'])/'processed'/name).convert('RGB').resize((640,h)),dtype=np.float32)/255
            images=[ref];labels=['Mapping RGB'];row={'scene':scene,'image':name}
            for label,white,mode,near,far in [('current',False,'antialiased',.01,10000),('white_only',True,'antialiased',.01,10000),('native',True,'classic',.01/scale,1e6/scale)]:
                pkg=render_from_pose_gsplat(model,torch.tensor(c.pose_w2c,dtype=torch.float32,device='cuda'),c.fov_x,c.fov_y,640,h,
                    bg_color=torch.full((3,),float(white),device='cuda'),render_mode='RGB',rgb_only=True,rasterize_mode=mode,near_plane=near,far_plane=far)
                im=pkg['render'].clamp(0,1).permute(1,2,0).cpu().numpy();del pkg
                row[label+'_psnr']=-10*math.log10(max(float(np.mean((im-ref)**2)),1e-15));images.append(im);labels.append(label)
            panel=Image.new('RGB',(2560,h+28));draw=ImageDraw.Draw(panel)
            for col,(im,label) in enumerate(zip(images,labels)):
                panel.paste(Image.fromarray(np.round(im*255).astype(np.uint8)),(col*640,28));draw.text((col*640+5,6),label,fill='white')
            panel.save(a.output/f'{scene}_{index}.png');rows.append(row);print(row,flush=True)
        reports.extend(rows);del model;torch.cuda.empty_cache()
        (a.output/'report.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
