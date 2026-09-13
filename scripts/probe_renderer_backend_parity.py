"""Read-only mapping-render parity for official 2DGS versus gsplat."""
import argparse
import json
from pathlib import Path
import math
import torch
from common.hashing import sha256_file
from data.datasets import ColmapDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['native','gsplat'],required=True)
    for name in ['dataset','prior','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--views',type=int,default=12)
    p.add_argument('--near-plane',type=float,default=.01)
    p.add_argument('--image-names',nargs='+')
    p.add_argument('--longest-edge',type=int,default=0)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    all_cameras=ColmapDataset(args.dataset,images='processed').split('mapping')
    cameras=[all_cameras[i] for i in torch.linspace(0,len(all_cameras)-1,min(args.views,len(all_cameras))).long()]
    if args.image_names:
        by_name={c.image_name:c for c in all_cameras}
        cameras=[by_name[n] for n in args.image_names]
    if args.longest_edge:
        from dataclasses import replace
        cameras=[replace(c,width=round(c.width*args.longest_edge/max(c.width,c.height)),
                         height=round(c.height*args.longest_edge/max(c.width,c.height))) for c in cameras]
    if args.backend=='native':
        from scene.gaussian_model import GaussianModel
        from diff_surfel_rasterization import GaussianRasterizationSettings,GaussianRasterizer
        from data.cameras import _projection
        model=GaussianModel(3);model.load_ply(str(args.prior))
    else:
        from priors.models import GaussianModel2D
        from priors.rendering import render_from_pose_gsplat
        model=GaussianModel2D(3);model.load_ply(args.prior,loc_feature_dim=0);model.cuda().eval()
    output=[]
    for camera in cameras:
        pose=torch.from_numpy(camera.pose_w2c).float().cuda()
        with torch.no_grad():
            if args.backend=='native':
                view=pose.T.contiguous();proj=_projection(.01,100,camera.fov_x,camera.fov_y).T.cuda()
                settings=GaussianRasterizationSettings(image_height=camera.height,image_width=camera.width,
                    tanfovx=math.tan(camera.fov_x/2),tanfovy=math.tan(camera.fov_y/2),bg=torch.zeros(3,device='cuda'),
                    scale_modifier=1.,viewmatrix=view,projmatrix=view@proj,sh_degree=3,campos=view.inverse()[3,:3],prefiltered=False,debug=False)
                raster=GaussianRasterizer(raster_settings=settings)
                rgb,_,_=raster(means3D=model.get_xyz,means2D=torch.zeros_like(model.get_xyz),shs=model.get_features,
                              colors_precomp=None,opacities=model.get_opacity,scales=model.get_scaling,rotations=model.get_rotation,cov3D_precomp=None)
            else:
                rgb=render_from_pose_gsplat(model,pose,camera.fov_x,camera.fov_y,camera.width,camera.height,
                    bg_color=torch.zeros(3,device='cuda'),render_mode='RGB',rgb_only=True,near_plane=args.near_plane)['render'][:3]
        output.append({'name':camera.image_name,'rgb':rgb.clamp(0,1).cpu()})
        print(args.backend,camera.image_name,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'backend':args.backend,'dataset':str(args.dataset),'prior_sha256':sha256_file(args.prior),
                'uses_source_mapping_rgb':False,'uses_test_queries':False,'black_background':True,
                'requested_near_plane':args.near_plane,'views':output},args.output)


if __name__=='__main__':main()
