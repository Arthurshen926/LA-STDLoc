"""Summarize backend parity without selecting pose parameters from test."""
import argparse
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.hashing import sha256_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    output=args.root/'summary.json'
    if output.exists():raise FileExistsError(output)
    native=torch.load(args.root/'OldHospital_native.pt',map_location='cpu',weights_only=False)
    rows=[];baseline=None
    for name in ['OldHospital_gsplat.pt','OldHospital_gsplat_near02.pt']:
        other=torch.load(args.root/name,map_location='cpu',weights_only=False)
        if other['prior_sha256']!=native['prior_sha256']:raise ValueError('different prior')
        cases=[]
        for a,b in zip(native['views'],other['views']):
            if a['name']!=b['name']:raise ValueError('different cameras')
            diff=a['rgb']-b['rgb']
            cases.append({'name':a['name'],'mae':float(diff.abs().mean()),'psnr':float(-10*torch.log10(diff.square().mean()))})
        rows.append({'arm':name,'artifact_sha256':sha256_file(args.root/name),'cases':cases})
        if baseline is None:baseline=other
    worst=min(range(len(rows[0]['cases'])),key=lambda i:rows[0]['cases'][i]['psnr'])
    a,b=native['views'][worst]['rgb'],baseline['views'][worst]['rgb']
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for axis,value,title in zip(ax,[a.permute(1,2,0),b.permute(1,2,0),(a-b).abs().mean(0)],['Official 2DGS','gsplat (same PLY/pose)','Mean absolute RGB difference']):
        if value.ndim==2:axis.imshow(value.numpy(),vmin=0,vmax=.1,cmap='inferno')
        else:axis.imshow(value.numpy())
        axis.set_title(title);axis.axis('off')
    fig.suptitle(native['views'][worst]['name']);fig.tight_layout();fig.savefig(args.root/'worst_view.png',dpi=140);plt.close(fig)
    output.write_text(json.dumps({'schema':'anygsloc_backend_parity_summary_v1','uses_source_mapping_rgb':False,'uses_test_queries':False,
                                  'prior_sha256':native['prior_sha256'],'native_sha256':sha256_file(args.root/'OldHospital_native.pt'),'arms':rows},indent=2))


if __name__=='__main__':main()
