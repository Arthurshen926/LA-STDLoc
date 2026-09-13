"""Assemble fixed-camera scientific renders; no generated image content."""
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    root=Path('/mnt/pool/sqy/anygsloc_stmary_canopy_comparison_20260910_v3')
    matcha=Path('/mnt/pool/sqy/anygsloc_stmary_canopy_comparison_20260910_v2')
    panels=[]
    for index in [632,633,634,408]:
        original=Image.open(matcha/f'{index:05d}_comparison.png').convert('RGB')
        ims=[original.crop((0,32,640,392)), Image.open(matcha/f'{index:05d}_matcha.png'),
             Image.open(root/f'vanilla2dgs_retry/{index:05d}_render.png'),
             Image.open(root/f'vanilla3dgs/{index:05d}_render.png'),original.crop((1280,32,1920,392))]
        labels=['Reference','MAtCHA 30k','Vanilla 2DGS 30k','Vanilla 3DGS 30k','G4Splat v310/800' if index!=408 else 'G4Splat v113/2k']
        for cropped in [False,True]:
            panel=Image.new('RGB',(640*5,392));draw=ImageDraw.Draw(panel)
            for i,(im,label) in enumerate(zip(ims,labels)):
                if cropped:im=im.crop((179,43,576,316)).resize((640,360))
                panel.paste(im,(i*640,32));draw.text((i*640+8,8),f'{index} | {label}',fill='white')
            panel.save(root/f'{index:05d}_{"crop" if cropped else "comparison"}.png')
            if not cropped:panels.append(panel)
    grid=Image.new('RGB',(3200,392*4))
    for i,panel in enumerate(panels):grid.paste(panel,(0,i*392))
    grid.save(root/'comparison_grid.png')
    (root/'comparison_notes.json').write_text(json.dumps({'indices':[632,633,634,408],
      'same_camera_same_resolution':True,'background':'black for all three Gaussian priors',
      'not_controlled_training_comparison':'MAtCHA used 512 optimization views; Vanilla used full mapping. Historical G4 settings differ.',
      'source_reports':[str(matcha/'report.json'),str(root/'vanilla2dgs_retry/report.json'),str(root/'vanilla3dgs/report.json')]},indent=2))
    print(root/'comparison_grid.png')


if __name__=='__main__':main()
