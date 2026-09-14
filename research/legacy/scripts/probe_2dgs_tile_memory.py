"""Full-resolution tile-size parity and memory probe, mapping views only."""
import argparse
import json
from pathlib import Path
import torch
from data.datasets import ColmapDataset
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    m=GaussianModel2D(3);m.load_ply('/mnt/pool/sqy/anygsloc_three_gpu_20260910/priors/ShopFacade/vanilla_2dgs/native/point_cloud/iteration_30000/point_cloud.ply',loc_feature_dim=0)
    cameras=ColmapDataset('/mnt/pool/sqy/Cambridge_stdloc/ShopFacade',images='processed').split('mapping')
    rows=[]
    for index in [0,24,25,26,27,28,29,30]:
        c=cameras[index]; outputs={};row={'index':index,'image':c.image_name,'width':c.width,'height':c.height}
        for tile in [16,32]:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            try:
                out=render_from_pose_gsplat(m,torch.from_numpy(c.pose_w2c).cuda().float(),c.fov_x,c.fov_y,c.width,c.height,
                    bg_color=torch.zeros(3,device='cuda'),render_mode='RGB+ED',rgb_only=True,tile_size=tile)
                outputs[tile]={k:out[k].cpu() for k in ['render','depth','rend_alpha']}
                row[str(tile)]={'peak_bytes':torch.cuda.max_memory_allocated()};del out
            except torch.cuda.OutOfMemoryError:row[str(tile)]={'oom':True}
        if len(outputs)==2:
            row['max_abs_diff']={k:float((outputs[16][k]-outputs[32][k]).abs().max()) for k in outputs[16]}
            row['mean_abs_diff']={k:float((outputs[16][k]-outputs[32][k]).abs().mean()) for k in outputs[16]}
        rows.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
