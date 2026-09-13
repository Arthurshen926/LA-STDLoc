"""Matched-camera feed-forward-prior panels on two otherwise idle GPUs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw


def main():
    pool=Path('/mnt/pool/sqy');root=pool/'anygsloc_stmary_feedforward_canopy_20260910'
    root.mkdir(exist_ok=False)
    prior=pool/'anygsloc_three_gpu_20260910/priors/StMarysChurch'
    jobs=[('long_lrm_raw','Long-LRM raw',prior/'long_lrm/world.ply',0),
          ('long_lrm_1k','Long-LRM +1k',prior/'long_lrm/ft1000/point_cloud/iteration_1000/point_cloud.ply',0),
          ('i3dgs_raw','i3DGS raw',prior/'i3dgs/world.ply',1),
          ('i3dgs_1k','i3DGS +1k',prior/'i3dgs/ft1000/point_cloud/iteration_1000/point_cloud.ply',1),
          ('long_lrm_2epochs','Long-LRM +2974 (extra budget)',pool/'anygsloc_long_lrm_two_epoch_gpu2_control_20260910/priors/StMarysChurch/long_lrm/ft_two_epochs/point_cloud/iteration_2974/point_cloud.ply',0)]
    def lane(gpu):
        for key,label,path,assigned in jobs:
            if assigned!=gpu:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),LD_LIBRARY_PATH='/root/miniconda3/envs/g4splat/lib',
                     PATH='/root/miniconda3/envs/g4splat/bin:/usr/local/cuda/bin:/usr/bin:/bin',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            with (root/f'{key}.log').open('w') as log:
                subprocess.run(['/root/miniconda3/envs/g4splat/bin/python','-m','scripts.render_stmary_canopy_comparison',
                    '--output',str(root/key),'--prior',str(path),'--gaussian-type','3dgs','--label',label],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('rendered',key,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lane,[0,1]))
    old=pool/'anygsloc_stmary_canopy_comparison_20260910_v3'
    panels=[]
    for index in [632,633,634,408]:
        original=Image.open(old/f'{index:05d}_comparison.png').convert('RGB')
        items=[('Reference',original.crop((0,32,640,392)))]+[(label,Image.open(root/key/f'{index:05d}_render.png')) for key,label,_,_ in jobs]
        for cropped in [False,True]:
            panel=Image.new('RGB',(1920,784));draw=ImageDraw.Draw(panel)
            for ordinal,(label,im) in enumerate(items):
                x=(ordinal%3)*640;y=(ordinal//3)*392
                if cropped:im=im.crop((179,43,576,316)).resize((640,360))
                panel.paste(im,(x,y+32));draw.text((x+8,y+8),label,fill='white')
            panel.save(root/f'{index:05d}_{"crop" if cropped else "comparison"}.png')
            if not cropped:panels.append(panel)
        # Full method lineup as a readable 3x3 sheet.
        all_items=[('Reference',original.crop((0,32,640,392))),('MAtCHA',original.crop((640,32,1280,392))),
          ('Vanilla 2DGS',original.crop((1280,32,1920,392))),('Vanilla 3DGS',original.crop((1920,32,2560,392)))]
        all_items.extend((label,Image.open(root/key/f'{index:05d}_render.png')) for key,label,_,_ in jobs[:4])
        all_items.append(('Historical G4Splat v310/800' if index!=408 else 'Historical G4Splat v113/2k',original.crop((2560,32,3200,392))))
        sheet=Image.new('RGB',(1920,1176));draw=ImageDraw.Draw(sheet)
        for ordinal,(label,im) in enumerate(all_items):
            x=(ordinal%3)*640;y=(ordinal//3)*392;sheet.paste(im,(x,y+32));draw.text((x+8,y+8),label,fill='white')
        sheet.save(root/f'{index:05d}_all_methods.png')
    grid=Image.new('RGB',(1920,784*4))
    for i,panel in enumerate(panels):grid.paste(panel,(0,i*784))
    grid.save(root/'feedforward_grid.png')
    (root/'report.json').write_text(json.dumps({'indices':[632,633,634,408],
        'methods':[{'key':k,'label':l,'prior':str(p),'report':str(root/k/'report.json')} for k,l,p,_ in jobs],
        'render_settings':'same 640x360 camera, black background, gsplat antialiased for all feedforward variants',
        'caveats':['Long-LRM native training used white/classic; these are harmonized pipeline renders, not native-renderer quality scores.',
                   'MAtCHA and feed-forward priors have different reconstruction view budgets.',
                   'G4Splat panels are historical, not asserted to be latest.',
                   'The +2974 condition has extra optimization budget and is not the +1k condition.']},indent=2))
    print('COMPLETE',root,flush=True)


if __name__=='__main__':main()
