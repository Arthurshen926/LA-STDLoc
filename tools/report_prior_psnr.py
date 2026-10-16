"""Package completed PSNR measurements without editing the paper or figures."""
from pathlib import Path
import json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1];O=ROOT/'artifacts/prior_psnr_20261008'
d=json.loads((O/'summary.json').read_text());keys=d['prior_order'];scenes=d['scene_order']
native=json.loads((O/'production_renderer_verification.json').read_text());assert native['status']=='passed'
assert d['n_rendered_comparisons']==9590 and d['n_test_images']==1918
cpu_checks=[]
for scene in scenes:
    for key in keys:
        job=O/'jobs'/f'{scene}__{key}'
        rows=[json.loads(x) for x in (job/'records.jsonl').read_text().splitlines()]
        first=next(r for r in rows if r['index']==0)
        err=abs(first['mse']-first['independent_CPU_mse']);assert err<1e-12
        assert all(r['tile_size']==16 for r in rows)
        assert all(np.isfinite(r['psnr_db']) and np.isfinite(r['mse']) for r in rows)
        cpu_checks.append({'scene':scene,'prior':key,'CPU_GPU_MSE_difference':err})

names={'StMarysChurch':"St. Mary's Church",'OldHospital':'Old Hospital','GreatCourt':'Great Court','ShopFacade':'Shop Facade','KingsCollege':"King's College"}
lines=['Gaussian prior PSNR — completed 2026-10-08',
       '25 / 25 scene-prior combinations; 1,918 common official test views per prior; 9,590 full-resolution renders.',
       '1920 x 1080; same rectified RGB / calibration / GT poses; full RGB image, no masks or fitted color/exposure.',
       'PSNR is computed per image in dB, then averaged per scene. The macro row gives all five scenes equal weight.',
       '', 'Scene                    N      MAtCha     2DGS     3DGS   i3DGS raw   i3DGS post-optimized']
for s in scenes:
    lines.append(f'{names[s]:24s} {d["scene_view_counts"][s]:4d}  '+'  '.join(f'{d["scene_means"][s][k]:9.4f}' for k in keys))
lines.append(f'{"Five-scene macro":24s} 1918  '+'  '.join(f'{d["five_scene_macro_psnr_db"][k]:9.4f}' for k in keys))
lines.append(f'{"Pooled image mean":24s} 1918  '+'  '.join(f'{d["pooled_image_mean_psnr_db"][k]:9.4f}' for k in keys))
delta=d['five_scene_macro_psnr_db']['i3dgs_1k']-d['five_scene_macro_psnr_db']['i3dgs_raw']
assert all(d['scene_means'][s]['i3dgs_1k']>d['scene_means'][s]['i3dgs_raw'] for s in scenes)
lines += ['',f'i3DGS post-optimization improves PSNR in all five scenes; scene-equal mean gain {delta:.4f} dB.',
          'MAtCha, 2DGS and 3DGS have similar five-scene mean PSNR (14.48–14.50 dB); avoid assigning significance to their tiny macro differences.',
          '', 'Checks: all views present for every prior; per-image metric recomputed from MSE; all 25 first-view CPU/GPU MSE checks passed;',
          'production-renderer cross-checks passed for native 2DGS and 3DGS; original model hashes unchanged; no OOM retry was used.',
          '', 'E4-ready scene means are in e4_psnr_rows.json. These evaluate the corresponding input priors, not the common pose-visualization model.',
          'The paper and figure artwork are not edited by this experiment.',
          '', 'Reproduce: /root/miniconda3/envs/g4splat/bin/python tools/run_prior_psnr.py',
          'Files: manifest.json (frozen inputs/protocol); per_image_psnr.csv; scene_psnr.csv; summary.json; verification.json;',
          'production_renderer_verification.json; jobs/<scene>__<prior>/{records.jsonl,summary.json}; worker_*.log.']
(O/'results.txt').write_text('\n'.join(lines)+'\n')
e4={'label':'PSNR — scene mean, dB','split':'All official test views of the indicated scene','resolution':[1920,1080],
    'keys':keys,'labels':[d['prior_labels'][k] for k in keys],
    'scenes':{s:{'n_views':d['scene_view_counts'][s],'PSNR_dB':d['scene_means'][s],
                 'display_values_2dp':[f'{d["scene_means"][s][k]:.2f}' for k in keys]} for s in ['StMarysChurch','ShopFacade']},
    'global_values':{'label':'Mean PSNR over five scenes (dB)','aggregation':'equal-weight mean of per-scene mean per-image PSNR',
                     'PSNR_dB':d['five_scene_macro_psnr_db']},'source':'summary.json',
    'figure_integration_done':False}
(O/'e4_psnr_rows.json').write_text(json.dumps(e4,indent=2,ensure_ascii=False)+'\n')
latex=['% Scene & test views & MAtCha & 2DGS & 3DGS & i3DGS Raw & i3DGS Post-optimized']
for s in scenes:
    latex.append(names[s]+' & '+str(d['scene_view_counts'][s])+' & '+' & '.join(f'{d["scene_means"][s][k]:.2f}' for k in keys)+r' \\')
latex.append('Five-scene mean & 1918 & '+' & '.join(f'{d["five_scene_macro_psnr_db"][k]:.2f}' for k in keys)+r' \\')
(O/'psnr_table_rows.tex').write_text('\n'.join(latex)+'\n')
verification=json.loads((O/'verification.json').read_text())
verification.update(first_view_CPU_GPU_checks=cpu_checks,production_renderer_cross_checks=native['checks'],
                    all_rows_finite=True,all_render_tile_sizes_16=True,figure_integration_done=False)
(O/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
files=[p for p in O.rglob('*') if p.is_file() and p.name!='output_manifest.sha256.json']
files += [ROOT/'tools'/n for n in ['evaluate_prior_psnr.py','run_prior_psnr.py','verify_prior_psnr_renderer.py','report_prior_psnr.py']]
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(O/'output_manifest.sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('\n'.join(lines[:16]))
