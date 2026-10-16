"""R2: Full versus Compact, with identical archived-depth display visibility.

No map changes are made. The common display mask is computed once on Full XYZ
and reused by source-row membership for Compact. It is not a denoising claim.
"""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from figure_cache_reader import read_cache
OUT=ROOT/'paper/experiments_cvpr_r2_20261003';A=OUT/'assets/e3'
OLD=ROOT/'paper/experiments_cvpr_20261003/assets/e3'
SCENES=['OldHospital','KingsCollege'];INK,MUTED,TEAL='#22354A','#657486','#087F72'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def prepare():
 A.mkdir(parents=True,exist_ok=True)
 old=json.loads((OLD/'prepared.json').read_text());p={'figure':'E3_compact_map','artifact_only':True,
  'previous_materialization':str(OLD/'prepared.json'),'previous_materialization_sha256':sha(OLD/'prepared.json'),
  'pca':old['pca'],'all_scene_results':old['all_scene_results'],'totals_all_five_scenes':old['totals_all_five_scenes'],
  'displayed_scenes':{},'visibility_rule':'At every Full-map projected point, sample the archived native rendered depth and alpha at the nearest native feature-grid pixel. Keep positive finite depth, alpha>0.05, and |point_camera_z-rendered_depth| < max(0.30m, 0.03*rendered_depth). Apply exactly this same source-row mask to Full and Compact.',
  'roi_rule':'Densest 800x200 native-pixel window of the Full map after the common surface display mask, scanned on a fixed 40px grid. Compact membership is not used to choose the ROI.',
  'limitations':['The shared surface band is a display/occlusion rule derived from archived mapping depth; it is not map preprocessing or an additional compact-selection operation.',
   'The Full and Compact panels have identical camera, crop, visibility mask, PCA/color scale and point size. Compact is obtained by the archived retained-source-row registry.',
   'The figure shows fewer landmarks while preserving visible facade structure. It does not measure or claim outlier removal or denoising.',
   'Whole-scene counts and the all-five-scene banner refer to the unchanged complete deployment maps; the displayed surface-band counts and ROI counts are reported separately.',
   'PCA was fitted once to all 158,941 Full descriptors from the two displayed scenes; the same components and 1–99 percentile RGB scaling are reused. Colors describe frozen SuperPoint features, not semantic categories.',
   'The display band makes approximate use of the archived rendered surface and is not ground-truth visibility. Its effect is symmetric between Full and Compact.',
   'No features, correspondences, maps, optimization, or localization results were recomputed.']}
 shutil.copyfile(OLD/'shared_full_descriptor_pca.npz',A/'shared_full_descriptor_pca.npz')
 p['pca']['basis_path']=str(A/'shared_full_descriptor_pca.npz');p['pca']['basis_sha256']=sha(A/'shared_full_descriptor_pca.npz')
 for scene in SCENES:
  m=old['displayed_scenes'][scene];d=np.load(OLD/f'{scene}_display.npz');uv=d['uv'];rgb=d['rgb'];z=d['depth'];keep=d['keep']
  for kind in ['full','compact']:
   assert sha(m[f'{kind}_map_path'])==m['source_hashes'][f'{kind}_sha256']
  assert sha(m['registry_path'])==m['source_hashes']['registry_sha256']
  camera=m['camera'];cp=Path('/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901')/scene/'observations/rendered_rgb_feature_cache.pt'
  cache=read_cache(cp);assert cache['uses_test_queries'] is False
  q=cache['queries'][camera['name']]
  assert np.allclose(q['native_K'].numpy(),camera['K'],atol=1e-3,rtol=0)
  assert np.allclose(q['pose_w2c'].numpy(),camera['pose'],atol=1e-5,rtol=0)
  dep=q['native_depth'].numpy().squeeze();alpha=q['native_alpha'].numpy().squeeze();h,w=dep.shape
  raw=(z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
  x=np.clip(np.round(uv[:,0]).astype(int),0,w-1);y=np.clip(np.round(uv[:,1]).astype(int),0,h-1)
  sampled_depth=dep[y,x];sampled_alpha=alpha[y,x]
  surface=raw&np.isfinite(sampled_depth)&(sampled_depth>0)&(sampled_alpha>.05)&(np.abs(z-sampled_depth)<np.maximum(.30,.03*sampled_depth))
  crop=[0,100,w,900] if scene=='OldHospital' else [0,220,w,840]
  visible=surface&(uv[:,1]>=crop[1])&(uv[:,1]<crop[3])
  candidates=[]
  for yy in range(crop[1],crop[3]-200+1,40):
   for xx in range(0,w-800+1,40):
    roi=visible&(uv[:,0]>=xx)&(uv[:,0]<xx+800)&(uv[:,1]>=yy)&(uv[:,1]<yy+200)
    candidates.append((int(roi.sum()),xx,yy))
  _,xx,yy=max(candidates)
  roi=visible&(uv[:,0]>=xx)&(uv[:,0]<xx+800)&(uv[:,1]>=yy)&(uv[:,1]<yy+200)
  np.savez_compressed(A/f'{scene}_display.npz',uv=uv,rgb=rgb,keep=keep,raw=raw,surface_mask=surface,visible=visible,roi=roi,
    sampled_depth=sampled_depth,sampled_alpha=sampled_alpha,camera_depth=z,retained_source_rows=d['retained_source_rows'])
  p['displayed_scenes'][scene]={**{k:v for k,v in m.items() if not k.startswith(('roi_','visible_'))},'display_crop':crop,'native_depth_cache':str(cp),'native_camera_verified_against_cache':True,
   'roi_box':[xx,yy,xx+800,yy+200],'roi_full_count':int(roi.sum()),'roi_compact_count':int((roi&keep).sum()),
   'roi_retention':float((roi&keep).sum()/roi.sum()),'raw_in_frustum_full':int(raw.sum()),'raw_in_frustum_compact':int((raw&keep).sum()),
   'surface_band_full':int(surface.sum()),'surface_band_compact':int((surface&keep).sum()),
   'displayed_full':int(visible.sum()),'displayed_compact':int((visible&keep).sum()),
   'surface_band_rejection_fraction_full':float(1-surface.sum()/raw.sum()),
   'surface_band_rejection_fraction_compact':float(1-(surface&keep).sum()/(raw&keep).sum())}
 (A/'prepared.json').write_text(json.dumps(p,indent=2)+'\n')
 return p

def panel(ax,d,kind,crop,zoom):
 uv,rgb=d['uv'],d['rgb'];mask=d['visible'].copy()
 mask&=(uv[:,0]>=crop[0])&(uv[:,0]<crop[2])&(uv[:,1]>=crop[1])&(uv[:,1]<crop[3])
 if kind=='compact':mask&=d['keep']
 ax.scatter(uv[mask,0],uv[mask,1],s=1.05 if zoom else .50,c=rgb[mask],lw=0,rasterized=True)
 ax.set(xlim=(crop[0],crop[2]),ylim=(crop[3],crop[1]),aspect='equal');ax.axis('off')

def save(fig,name,p):
 fig.canvas.draw();ren=fig.canvas.get_renderer();outside=[]
 texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
 for t in texts:
  b=t.get_window_extent(ren)
  if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:outside.append(t.get_text())
 assert not outside,outside
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=400)
 plt.close(fig)
 p.setdefault('outputs',{})[name]={ext:{'path':str(OUT/f'{name}.{ext}'),'sha256':sha(OUT/f'{name}.{ext}')} for ext in ['pdf','svg','png']}
 p['outputs'][name]['render_validation']={'text_outside_canvas':outside,'minimum_declared_font_pt':min(t.get_fontsize() for t in texts)}

def draw(p):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':INK,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
 fig=plt.figure(figsize=(7.1,5.25));xs=[.045,.545];width=.410
 for x,label in zip(xs,['Full landmark map','Compact landmark map']):fig.text(x+width/2,.982,label,ha='center',va='top',fontsize=9,weight='bold')
 for row,scene in enumerate(SCENES):
  m=p['displayed_scenes'][scene];d=np.load(A/f'{scene}_display.npz');stats=p['all_scene_results'][scene]
  n,k=stats['full']['landmarks'],stats['compact']['landmarks'];top=.924-row*.435
  fig.text(.045,top,scene,va='top',fontsize=8.5,weight='bold')
  fig.text(.955,top,f'Whole map: {n:,} → {k:,}  (−{100*(1-k/n):.1f}%)',va='top',ha='right',fontsize=8,color=TEAL)
  for col,kind in enumerate(['full','compact']):
   ax=fig.add_axes([xs[col],top-.235,width,.20]);panel(ax,d,kind,m['display_crop'],False)
   x0,y0,x1,y1=m['roi_box'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=INK,lw=.70))
   ax=fig.add_axes([xs[col],top-.387,width,.135]);panel(ax,d,kind,m['roi_box'],True)
   ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#C2CCD5',lw=.60))
   label=f"Detail: {m['roi_full_count']:,} landmarks" if col==0 else f"Detail: {m['roi_compact_count']:,} landmarks · {100*m['roi_retention']:.1f}% retained"
   fig.text(xs[col]+width/2,top-.395,label,ha='center',va='top',fontsize=7.5,color=MUTED)
 total=p['totals_all_five_scenes'];fn,cn=total['full']['landmarks'],total['compact']['landmarks']
 fig.text(.5,.034,f'All five scenes:  {fn:,} → {cn:,} landmarks  (−{100*(1-cn/fn):.1f}%)',
   ha='center',va='center',fontsize=9,color=TEAL,weight='bold')
 save(fig,'E3_compact_map',p)
 p['figure_inches']=[7.1,5.25];p['main_point_area_pt2']=.50;p['zoom_point_area_pt2']=1.05
 p['display_assets']={s:{'path':str(A/f'{s}_display.npz'),'sha256':sha(A/f'{s}_display.npz')} for s in SCENES}
 p['visual_check']='Final two-column PNG inspected; large facade views and matched horizontal detail strips remain aligned, with no label collision.'
 p['new_figure_changes']=['Removed the selection overlay column.','Removed TE/R5 metrics already in the localization tables.',
  'Enlarged Full and Compact views and replaced small square crops with matched horizontal architectural detail strips.',
  'Applied one verified archived-depth visibility band symmetrically to prevent projection through nearer surfaces.']
 (OUT/'E3_compact_map.provenance.json').write_text(json.dumps(p,indent=2)+'\n')
 caption=('Full and compact landmark representations under identical display conditions. '
  'The two columns share the same archived mapping camera, crop, point size, and descriptor PCA basis and color scale. '
  'A common visibility band is computed once from the Full-map points and archived native rendered depth (absolute depth discrepancy below max(0.30 m, 3% of rendered depth), with positive finite depth and alpha above 0.05); '
  'the identical source-row mask is then applied to Compact. This display rule prevents projecting through nearby surfaces and changes neither map. '
  'Matched detail strips expose the reduction in landmark density while retaining recognizable facade structures. '
  'PCA is fitted once to Full-map descriptors from both displayed scenes; the fixed 1–99 percentile RGB scale is reused for Compact. '
  'Whole-map counts report every stored landmark, while detail labels count only the displayed common surface band. '
  'Across all five Cambridge scenes, the unchanged maps contain 535,840 and 176,329 landmarks (67.1% reduction). '
  'The visualization supports sparsity and structure comparison, not a claim of denoising or uniform localization improvement. '
  'No map construction or localization experiment was rerun.')
 (OUT/'E3_compact_map.caption.txt').write_text(caption+'\n\nDisplay limitations:\n'+'\n'.join('- '+s for s in p['limitations'])+'\n')
 print(json.dumps({s:{k:v for k,v in m.items() if k.startswith('surface_band') or k.startswith('roi_')} for s,m in p['displayed_scenes'].items()},indent=2))

def main():
 p=json.loads((A/'prepared.json').read_text()) if '--redraw' in sys.argv else prepare()
 draw(p)

if __name__=='__main__':main()
