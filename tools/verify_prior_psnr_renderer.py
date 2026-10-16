"""Cross-check the evaluator against the existing production RGB renderer."""
from pathlib import Path
import json, math, os, sys
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from anygsloc.priors.models import FrozenGaussianModel
from anygsloc.priors.rendering import render_from_pose_gsplat

O=ROOT/'artifacts/prior_psnr_20261008';m=json.loads((O/'manifest.json').read_text())
torch.set_num_threads(4)
checks=[]
for key in ['matcha','vanilla_3dgs']:
    scene='StMarysChurch';source=m['scenes'][scene]['priors'][key];v=m['scenes'][scene]['views'][0]
    model=FrozenGaussianModel(source['sh_degree'],device='cuda');model.load_ply(source['ply'],loc_feature_dim=0)
    K=np.asarray(v['K']);w,h=v['width'],v['height']
    with torch.inference_mode():
        result=render_from_pose_gsplat(model,torch.tensor(v['pose_w2c'],dtype=torch.float32,device='cuda'),
            2*math.atan(w/(2*K[0,0])),2*math.atan(h/(2*K[1,1])),w,h,
            bg_color=torch.zeros(3,device='cuda'),render_mode='RGB',rgb_only=True,rasterize_mode='antialiased')
        rgb=result['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
    gt=np.asarray(Image.open(v['path'])).astype(np.float64)/255.
    mse=float(np.mean((rgb.astype(np.float64)-gt)**2));psnr=-10*math.log10(mse)
    job=O/'jobs'/f'{scene}__{key}';record=json.loads((job/'records.jsonl').read_text().splitlines()[0])
    assert abs(mse-record['mse'])<1e-12
    quantized=np.uint8(np.round(rgb*255))
    assert np.array_equal(quantized,np.array(Image.open(job/'first_view_render.png')))
    checks.append({'scene':scene,'prior':key,'name':v['name'],'production_renderer':'anygsloc.priors.rendering.render_from_pose_gsplat',
                   'mse_difference':abs(mse-record['mse']),'PSNR_difference_dB':abs(psnr-record['psnr_db']),
                   'saved_render_pixels_exact_match':True})
    del model,result;torch.cuda.empty_cache()
(O/'production_renderer_verification.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n')
print(json.dumps(checks,indent=2))
