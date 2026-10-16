"""Editable spatial mechanism plates; conceptual geometry is explicitly labelled.

Real image assets and map coordinates remain separate from schematic operations.
No experiment, map, camera estimate, or descriptor is modified by this tool.
"""
from pathlib import Path
import sys,json,base64,html,shutil,hashlib,io
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/root/.cache/anygsloc_figure_deps')
import cairosvg
OUT=ROOT/'paper/figures_spatial_final';OUT.mkdir(exist_ok=True)
A=ROOT/'paper/figures_v5/assets';PREV=ROOT/'paper/figures_submission_candidate'
INK='#243349';MUTED='#66788a';BLUE='#397eaf';TEAL='#187b70';GOLD='#b78343';RED='#bd5954';LIGHT='#dce4eb'
S=[];MAN={}

def add(s):S.append(s)
def text(x,y,s,size=26,color=INK,bold=False,anchor='start'):
    add(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{600 if bold else 400}" fill="{color}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
def line(a,b,color=MUTED,width=2,dash='',arrow=False):
    add(f'<path d="M{a[0]},{a[1]} L{b[0]},{b[1]}" fill="none" stroke="{color}" stroke-width="{width}" '+(f'stroke-dasharray="{dash}" ' if dash else '')+('marker-end="url(#arrow)" ' if arrow else '')+'/>')
def poly(points,fill='none',stroke=LIGHT,width=1.5,opacity=1):
    add('<polygon points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+f'" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>')
def rect(x,y,w,h,fill='white',stroke=LIGHT,r=0):add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
def dot(x,y,r=6,col=TEAL,fill=None):add(f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{col}" stroke-width="2" fill="{col if fill is None else fill}"/>')
def img(path,x,y,w,h):
    raw=base64.b64encode(Path(path).read_bytes()).decode();add(f'<image href="data:image/png;base64,{raw}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet"/>')
def plane(path,p,u,v,stroke=BLUE):
    """Affine image plane: real raster on an explicitly illustrative plane."""
    raw=base64.b64encode(Path(path).read_bytes()).decode();p=np.array(p);u=np.array(u);v=np.array(v)
    add(f'<image href="data:image/png;base64,{raw}" width="1" height="1" preserveAspectRatio="none" transform="matrix({u[0]} {u[1]} {v[0]} {v[1]} {p[0]} {p[1]})"/>')
    poly([p,p+u,p+u+v,p+v],stroke=stroke,width=2)
def begin(h,title,sub):
    S.clear();add(f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{h/10}mm" viewBox="0 0 1800 {h}">')
    add('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10" fill="#66788a"/></marker><radialGradient id="landmark"><stop offset="0" stop-color="#9ad2c7"/><stop offset="1" stop-color="#187b70"/></radialGradient></defs>')
    rect(0,0,1800,h,'white','white');text(40,48,title,33,bold=True);text(40,88,sub,24,MUTED)
def stage(x,y,letter,title,color,w):
    text(x,y,letter,27,color,True);text(x+35,y,title,29,INK,True);line((x,y+15),(x+w,y+15),color,2)
def finish(name,h,caption,limits):
    add('</svg>');p=OUT/(name+'.svg');p.write_text('\n'.join(S))
    cairosvg.svg2pdf(url=str(p),write_to=str(p.with_suffix('.pdf')))
    cairosvg.svg2png(url=str(p),write_to=str(p.with_suffix('.png')),output_width=3000)
    MAN[name]={'caption':caption,'limitations':limits}

def camera(c,p,u,v,col):
    c=np.array(c);p=np.array(p);u=np.array(u);v=np.array(v)
    for q in [p,p+u,p+v,p+u+v]:line(c,q,col,1.1)
    dot(*c,4,col)

def ray_diagram(x,y,scale=1):
    """Idealized three-camera geometry, NEVER a measured camera configuration."""
    add(f'<g transform="translate({x},{y}) scale({scale})">')
    # A lightly drawn ground plane supplies depth without a decorative 3D cube.
    poly([(35,355),(360,432),(592,317),(272,241)],'#f7f9fb',LIGHT)
    for t in [.25,.5,.75]:
        line((35+325*t,355+77*t),(272+320*t,241+76*t),LIGHT,1)
        line((35+237*t,355-114*t),(360+232*t,432-115*t),LIGHT,1)
    target=np.array([310.,66.])
    configs=[((24,231),(138,26),(-12,99),(8,391)),((237,269),(137,5),(8,94),(319,455)),((439,228),(130,-35),(27,95),(612,376))]
    for i,(p,u,v,c) in enumerate(configs):
        p=np.array(p);u=np.array(u);v=np.array(v);middle=p+.5*u+.5*v
        # Keep schematic pinhole rays collinear through the observation plane.
        c=middle+(middle-target)*.44
        camera(c,p,u,v,BLUE);line(c,target,TEAL,2)
        plane(A/f'StMarysChurch_geometry_{i}_crop.png',p,u,v)
        dot(*middle,6,TEAL,'white');line(middle,target,TEAL,2)
        text(p[0]+18,p[1]+145,f'view {i+1}',23,MUTED)
    dot(*target,12,TEAL,'url(#landmark)');text(332,63,'X',32,TEAL,True)
    # A displaced coordinate is an explicit conceptual contrast, not fake data.
    other=target+[-95,38];poly([other+[-7,-7],other+[7,-7],other+[7,7],other+[-7,7]],'white',GOLD,2)
    line(other,target,GOLD,1.5,'5 4');text(20,17,'render-derived coordinate',21,GOLD)
    line((168,25),other,GOLD,1)
    text(285,132,'observation rays',23,TEAL)
    add('</g>')

def descriptor(x,y,col=BLUE,n=6):
    # Equal-size glyph cells deliberately carry no invented feature values.
    for j in range(n):rect(x+j*17,y,14,27,'#e5eef5',col)

def evidence_stack(x,y,scale=1):
    add(f'<g transform="translate({x},{y}) scale({scale})">')
    for path,p,label,col in [(A/'StMarysChurch_validity_crop.png',(10,18),'Gaussian RGB',BLUE),
                            (A/'art_depth.png',(67,128),'Native depth',GOLD),
                            (A/'spatial_binary_mask.png',(124,238),'V2 valid support',TEAL)]:
        plane(path,p,(262,24),(-38,145),col);text(p[0]+290,p[1]+80,label,24,col)
    add('</g>')

def map_assets():
    """Orthographic 3D projection of real maps with exactly matched scale."""
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('/mnt/pool/sqy/anygsloc_observation_factorial_20260914/maps/OldHospital_v1_f1_c1')
    state=torch.load(root/'full/deployment_map.pt',map_location='cpu',weights_only=True)
    xyz=state['anchor_xyz'].numpy();idx=np.array(json.loads((root/'report.json').read_text())['retained_source_rows']);keep=np.zeros(len(xyz),bool);keep[idx]=True
    m=json.loads((ROOT/'paper/figures_v5/selection.json').read_text());view=next(v for v in m['views'] if v['scene']=='OldHospital' and 'thresholds' in v)
    pose=np.array(view['pose_w2c']);q=xyz@pose[:3,:3].T+pose[:3,3]
    k=np.array(view['native_K']);uv=q@k.T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8);h,w=view['native_input_hw']
    roi=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    # Shared source frustum and oblique view; never scale each map separately.
    az=np.radians(12);R=np.array([[np.cos(az),0,np.sin(az)],[0,-1,0],[-np.sin(az),0,np.cos(az)]])
    q=q@R.T;lo,hi=np.percentile(q[roi,:2],[.5,99.5],axis=0)
    for compact in [False,True]:
        fig=plt.figure(figsize=(5,3));ax=fig.add_axes([0,0,1,1])
        if not compact:ax.scatter(q[roi&~keep,0],q[roi&~keep,1],s=.20,c='#b3bfcb',alpha=.75,rasterized=True)
        ax.scatter(q[roi&keep,0],q[roi&keep,1],s=.20,c=TEAL,alpha=.85,rasterized=True)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');ax.axis('off')
        fig.savefig(A/('spatial_map_compact.png' if compact else 'spatial_map_full.png'),dpi=220,transparent=False,facecolor='white');plt.close(fig)
    (OUT/'map_projection.json').write_text(json.dumps({'scene':'OldHospital','source':str(root),'source_view':view['name'],'camera_pose':pose.tolist(),'view_matrix':R.tolist(),'full_count':len(xyz),'compact_count':len(idx),'point_size':.20,'same_scale':True,'all_points_displayed':False,'display_policy':'Shared source-camera frustum; full-map 0.5–99.5 percentile orthographic bounds applied to both; counts refer to entire maps.','roi_full':int(roi.sum()),'roi_compact':int((roi&keep).sum())},indent=2))

def overview():
    begin(1190,'AnyGSLoc','Rendering evidence, reconstructed landmarks, compact deployment')
    stage(40,147,'a','Validate observations',BLUE,490);stage(625,147,'b','Build localization landmarks',GOLD,560);stage(1300,147,'c','Compact selection',TEAL,455)
    evidence_stack(40,189,.75)
    text(40,558,'Native buffers → V2 predicate',27,BLUE,True)
    text(40,598,'Filter detected observations',26);text(40,634,'Not semantic segmentation',24,MUTED)
    line((547,356),(600,356),arrow=True)
    ray_diagram(620,186,.90)
    text(625,643,'Mutual match · epipolar check · tracks',23,MUTED)
    text(625,684,'Fixed-camera rays determine X',27,TEAL,True)
    descriptor(627,721);descriptor(655,758);line((777,748),(890,748),arrow=True)
    descriptor(918,733,TEAL);text(1050,755,'d',30,TEAL,True)
    text(625,823,'View-balanced, robust descriptor fusion',25)
    line((1206,355),(1270,355),arrow=True)
    img(A/'spatial_map_full.png',1290,198,465,225);text(1530,449,'73,241 candidate landmarks',25,MUTED,anchor='middle')
    line((1530,467),(1530,512),arrow=True)
    img(A/'spatial_map_compact.png',1290,517,465,225);text(1530,769,'16,589 selected landmarks',27,TEAL,True,'middle')
    text(1530,815,'Coverage + pose-information support',23,INK,anchor='middle')
    line((40,860),(1760,860),LIGHT,2,'7 7')
    text(40,908,'ONLINE',25,TEAL,True);img(A/'art_query.png',40,930,240,150)
    text(322,985,'Local features',29,bold=True);text(322,1026,'Frozen SuperPoint',23,MUTED)
    line((568,991),(635,991),arrow=True)
    text(680,985,'Sparse 2D–3D matching',29,bold=True);text(680,1026,'Deployed map: {X, d}',24,TEAL)
    line((1110,991),(1190,991),arrow=True)
    text(1240,985,'Robust PnP',29,bold=True);text(1480,985,'→ R, t',29,TEAL,True)
    text(1240,1026,'One pose-solver call',23,MUTED)
    add('<path d="M1665,836 L1665,905 L875,905 L875,950" fill="none" stroke="#187b70" stroke-width="2" marker-end="url(#arrow)"/>')
    text(900,1121,'No online renderer · No additional dense refinement',27,TEAL,anchor='middle')
    text(40,1163,'Spatial construction is schematic; image evidence: StMarysChurch. Map projections and query: OldHospital.',20,MUTED)
    finish('F02_overview',1190,'AnyGSLoc separates rendering primitives from localization landmarks. Native Gaussian-rendered evidence screens detected observations. Verified tracks determine fixed-camera ray geometry and view-balanced robust descriptors; compact selection retains matching coverage and pose-information support. Only XYZ and descriptors are deployed for single-stage sparse localization.',[
        'Camera arrangement and coordinate displacement in the spatial diagram are schematic, not measured geometry. Actual StMarysChurch rendered patches are texture references.','OldHospital maps use real XYZ, a shared source-camera frustum and orthographic view, identical point size and full-map 0.5–99.5 percentile display bounds; counts refer to entire maps.','The mask is the real full-resolution V2 predicate visualized on a schematic plane, not query semantic masking.'])

def construction():
    begin(1040,'From admissible observations to a localization landmark','A shared track links geometry and appearance; the Gaussian center is not the landmark definition')
    stage(40,147,'a','Geometric support',BLUE,780);stage(930,147,'b','View-balanced descriptor fusion',TEAL,815)
    ray_diagram(65,175,1.08)
    text(40,730,'Known cameras + checked multiview track',29,BLUE,True)
    text(40,771,'Epipolar-screened reciprocal matches → tracks',25)
    text(40,812,'Robust fixed-camera multi-ray triangulation → X',25)
    text(930,216,'Observations are grouped by mapping-view bin',26)
    for row,label in enumerate(['bin 1','bin 2','bin 3','bin 4']):
        y=266+row*100;rect(929,y-14,210,72,'#f6f8fa',LIGHT)
        descriptor(945,y,BLUE);text(1066,y+20,label,22,MUTED)
        line((1154,y+14),(1203,y+14),arrow=True);descriptor(1231,y,BLUE)
    text(960,684,'confidence-weighted',23,MUTED);text(1230,684,'bin prototypes',23,MUTED)
    # Medoid trimming is explicitly conceptual: this pattern is not measured.
    for row in range(4):
        y=280+row*100;line((1341,y),(1430,430),RED if row==3 else TEAL,1.7,'5 4' if row==3 else '')
    dot(1430,430,32,TEAL,'#e8f3ef');text(1430,440,'m',29,TEAL,True,'middle')
    text(1480,288,'medoid-guided',23,TEAL);text(1480,321,'selection',23,TEAL);line((1460,333),(1434,391),TEAL,1.4)
    line((1478,430),(1550,430),arrow=True);descriptor(1580,416,TEAL);text(1712,439,'d',32,TEAL,True)
    text(1450,552,'trim distant bins',24,RED);text(1430,603,'average + normalize',24,TEAL)
    text(930,736,'One descriptor per landmark',29,TEAL,True)
    text(930,779,'Not one descriptor per view; no learned adapter',24)
    line((40,849),(1760,849),LIGHT,2)
    for i in range(3):
        img(A/f'StMarysChurch_geometry_{i}_crop.png',45+i*250,879,150,126)
        text(210+i*250,934,f'view {i+1}',23,MUTED)
    text(950,909,'ACTUAL SUPPORT',22,BLUE,True)
    text(950,950,'Three recorded patches from one StMarysChurch track',23)
    text(950,986,'Spatial layout and descriptor glyphs above are schematic',21,MUTED)
    finish('F09_track_geometry_and_fusion',1040,'Track-level landmark construction. Known mapping cameras and verified image observations constrain a localization coordinate independently of the Gaussian primitive center. For descriptors, confidence-weighted view-bin prototypes are computed, bins most distant from the medoid are trimmed, and retained prototypes are averaged and normalized. The bottom row shows actual patches from one recorded track.',[
        'Illustrative cameras and rays explain dependence, not a metric rendering of the selected camera poses.','The displaced render-derived coordinate is schematic; measured coordinate comparisons remain in F04.','One rejected prototype glyph illustrates trimming, not a measured feature distribution or the exact number of rejected bins for this track.','No surface-normal fusion weights or LGCV module are claimed.'])

def selection():
    begin(950,'Compact selection preserves complementary constraints','Mapping-only selection: quality seeds, reference-view matching coverage, then pose support')
    for x,letter,title,col in [(40,'a','Quality seed',BLUE),(638,'b','Matching coverage',GOLD),(1235,'c','Pose-support completion',TEAL)]:stage(x,147,letter,title,col,505)
    # An oblique image plane has screen cells rather than a decorative 3D box.
    def grid(x,y):
        p=np.array([x,y]);u=np.array([340,45]);v=np.array([-60,210]);poly([p,p+u,p+u+v,p+v],'#f8fafb',LIGHT)
        for t in [.25,.5,.75]:line(p+t*u,p+t*u+v,LIGHT,1);line(p+t*v,p+t*v+u,LIGHT,1)
        return lambda a,b:p+a*u+b*v
    seeds=[(.23,.2),(.31,.22),(.39,.2),(.26,.34),(.34,.38),(.65,.6)];extras=[(.8,.18),(.18,.78),(.82,.82)]
    q=grid(92,215)
    for a,b in seeds:dot(*q(a,b),8,BLUE)
    text(40,533,'Start from reliable candidates',26,BLUE,True)
    text(40,575,'Quality order with coverage-aware stopping',23,MUTED)
    q=grid(690,215)
    for a,b in seeds:dot(*q(a,b),8,BLUE)
    for a,b in extras:dot(*q(a,b),9,GOLD)
    text(638,533,'Complete per-view matching coverage',25,GOLD,True)
    text(638,575,'Count distinct support in each reference view',23,MUTED)
    q=grid(1287,215)
    for a,b in seeds:dot(*q(a,b),8,BLUE)
    for a,b in extras:dot(*q(a,b),9,GOLD)
    for a,b,z in [(.5,.15,90),(.72,.75,135),(.15,.5,70)]:
        p=q(a,b);end=p+np.array([20,-z]);line(p,end,TEAL,1.7,'4 4');dot(*end,9,TEAL)
    text(1235,533,'Add complementary pose constraints',25,TEAL,True)
    text(1235,575,'Image cells · depth · 3D spatial diversity',23,MUTED)
    for x in [554,1153]:line((x,355),(x+55,355),arrow=True)
    line((40,620),(1760,620),LIGHT,2)
    img(A/'spatial_map_full.png',45,643,510,240);img(A/'spatial_map_compact.png',657,643,510,240)
    line((569,762),(627,762),arrow=True)
    text(300,919,'73,241 full landmarks',25,MUTED,anchor='middle');text(917,919,'16,589 selected landmarks',25,TEAL,True,'middle')
    text(1240,699,'ACTUAL OLDHOSPITAL MAP',22,BLUE,True)
    text(1240,752,'77.4% fewer landmarks',29,TEAL,True)
    text(1240,800,'Same projection and point size',24)
    text(1240,844,'No selector-induced FPS gain claimed',22,MUTED)
    finish('F10_compact_selection_mechanism',950,'Mapping-only compact selection: a quality-ordered core uses a matching-coverage stopping rule; per-query completion adds missing matching support, followed by pose-information completion with image-cell, depth and 3D spatial constraints. The top diagrams illustrate the roles of the stages; the bottom shows real OldHospital full and selected maps in an identical orthographic view.',[
        'Top-row candidate positions and selections are schematic, not an audited selector event trace or computed information-gain illustration.','77.4% is an OldHospital example, not the five-scene aggregate (67.1%).','Map panels share a source-camera frustum and full-map 0.5–99.5 percentile display bounds, with identical scale and marker size; counts refer to whole maps. Projection is not an occlusion test.','Compression does not imply lossless accuracy or measured latency improvement.'])

def main():
    old=json.loads((PREV/'manifest.json').read_text());MAN.update(old['figures'])
    for name in old['figures']:
        for ext in ['pdf','svg','png']:shutil.copyfile(PREV/(name+'.'+ext),OUT/(name+'.'+ext))
    mask=np.load(A/'StMarysChurch_validity_mask.npz')['valid'];Image.fromarray(mask.astype(np.uint8)*255).save(A/'spatial_binary_mask.png')
    map_assets();overview();construction();selection()
    manifest={'status':'complete_figure_set_candidate_for_author_review','figures':MAN,'inherited_sources':old['source_sha256'],
        'drawing_semantics':'All schematic camera/descriptor/selection layouts are labelled. Actual scientific images and map coordinates are not synthesized.',
        'references':{'ULF-Loc':'https://arxiv.org/html/2605.04730v1','STDLoc':'https://arxiv.org/html/2503.19358v1','DSAC':'https://arxiv.org/abs/1611.05705'}}
    sources=[Path(__file__),ROOT/'anygsloc/mapping/tracks.py',ROOT/'anygsloc/mapping/compact.py',ROOT/'anygsloc/mapping/selector.py',OUT/'map_projection.json']+list(A.glob('*_validity_replay.json'))
    manifest['current_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    captions=[];tex=[]
    for name,entry in sorted(MAN.items()):
        captions+=['## '+name,'',entry['caption'],'']+['- '+s for s in entry.get('limitations',[])]+['']
        tex.append('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures_spatial_final/'+name+'.pdf}\n\\caption{'+entry['caption'].replace('%','\\%').replace('&','\\&')+'}\n\\label{fig:'+name.lower()+'}\n\\end{figure*}\n')
    (OUT/'CAPTIONS.md').write_text('\n'.join(captions));(OUT/'figures.tex').write_text('\n'.join(tex))
    import pymupdf
    pdf=pymupdf.open();board=Image.new('RGB',(1800,550*((len(MAN)+1)//2)),'#edf1f4')
    for i,name in enumerate(sorted(MAN)):
        doc=pymupdf.open(OUT/(name+'.pdf'));pdf.insert_pdf(doc);doc.close()
        im=Image.open(OUT/(name+'.png')).convert('RGB');im.thumbnail((860,500));x=(i%2)*900+(900-im.width)//2;y=(i//2)*550+35+(500-im.height)//2;board.paste(im,(x,y));ImageDraw.Draw(board).text(((i%2)*900+20,(i//2)*550+12),name,fill=INK)
    pdf.save(OUT/'All_figures_review.pdf');pdf.close();board.save(OUT/'contact_sheet.jpg',quality=94)
    import xml.etree.ElementTree as ET
    qa={'figure_count':len(MAN),'formats':{},'new_vector_layout':{}}
    for name in MAN:
        ET.parse(OUT/(name+'.svg'))
        with Image.open(OUT/(name+'.png')) as im:im.verify()
        doc=pymupdf.open(OUT/(name+'.pdf'));assert len(doc)==1
        qa['formats'][name]='PDF, SVG and PNG readable'
        if name in ['F02_overview','F09_track_geometry_and_fusion','F10_compact_selection_mechanism']:
            page=doc[0];spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
            outside=[s['text'] for s in spans if not page.rect.contains(pymupdf.Rect(s['bbox']))]
            overlap=[]
            for i,s in enumerate(spans):
                for t in spans[i+1:]:
                    r=pymupdf.Rect(s['bbox']) & pymupdf.Rect(t['bbox'])
                    if not r.is_empty and r.get_area()>5:overlap.append([s['text'],t['text']])
            qa['new_vector_layout'][name]={'outside_page':outside,'text_overlap':overlap}
            assert not outside and not overlap,(name,outside,overlap)
        doc.close()
    (OUT/'QA.json').write_text(json.dumps(qa,indent=2))
    print('Spatial figure suite complete:',len(MAN),'figures',flush=True)

if __name__=='__main__':main()
