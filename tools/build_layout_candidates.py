"""Two vector overview compositions sharing verified mechanism assets."""
from pathlib import Path
import sys,json,base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import draw_mechanism_figures as d
import build_publication_overview as v
OUT=d.OUT;A=d.A

def raster(name,callback):
    fig=plt.figure(figsize=(3.2,2.4));ax=fig.add_axes([0,0,1,1]);callback(ax)
    fig.savefig(A/name,dpi=220,pad_inches=0,facecolor='white');plt.close(fig)

def main():
    m=json.loads((OUT/'selection.json').read_text());f=json.loads((OUT/'figure_manifest.json').read_text());hero=f['validity_hero'];geo=f['geometry_hero']
    raster('art_gaussians.png',lambda ax:d.gaussians(ax,hero))
    raster('art_validity.png',lambda ax:d.validity(ax,hero))
    raster('art_depth.png',lambda ax:d.buffers(ax,hero,'depth'))
    raster('art_structure.png',lambda ax:d.buffers(ax,hero,'structure'))
    mp=np.load(A/'OldHospital_selection_projection.npz');h,w=mp['hw']
    for compact in [False,True]:
        def drawmap(ax,compact=compact):
            uv=mp['uv'];keep=mp['keep'];good=mp['valid']
            if not compact:ax.scatter(uv[good&~keep,0],uv[good&~keep,1],s=.09,c='#c4cbd1',rasterized=True)
            ax.scatter(uv[good&keep,0],uv[good&keep,1],s=.09,c=v.TEAL,rasterized=True)
            ax.set(xlim=(0,w),ylim=(h,0),aspect='equal');ax.axis('off')
        raster('art_map_compact.png' if compact else 'art_map_full.png',drawmap)
    # Query image illustrates the input, not a new localization accuracy experiment.
    p=d.POOL/'anygsloc_protocol_v2_primary_20260914/runs/Cambridge/OldHospital/results.json'
    results=json.loads(p.read_text());ranked=sorted(results,key=lambda r:float(r['translation_error_cm']) if r['success'] else float('inf'))
    name=ranked[len(ranked)//2]['image_name'];query=d.POOL/'Cambridge_stdloc/OldHospital/processed'/name
    Image.open(query).convert('RGB').save(A/'art_query.png')
    v.ASSETS=A
    def begin(height):
        v.S.clear();v.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{height/10}mm" viewBox="0 0 1800 {height}">')
        v.add('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10" fill="#738393"/></marker></defs>')
        v.rect(0,0,1800,height,'white','white',0)
    def finish(name,height):
        v.add('</svg>');path=OUT/(name+'.svg');path.write_text('\n'.join(v.S))
        sys.path.insert(0,'/root/.cache/anygsloc_figure_deps');import cairosvg
        cairosvg.svg2pdf(url=str(path),write_to=str(path.with_suffix('.pdf')))
        cairosvg.svg2png(url=str(path),write_to=str(path.with_suffix('.png')),output_width=2700,output_height=round(height*1.5))
    def stage(x,y,number,title,color):
        v.text(x,y,number,20,color,700);v.text(x+38,y,title,26,v.INK,600);v.line(x,y+17,x+340,y+17,color,2)
    def fusion(x,y):
        # Descriptor glyphs encode data flow, not artificial numerical feature values.
        for j in range(3):
            v.rect(x,y+j*28,125,18,'#e5eef3',v.BLUE,2);v.text(x+140,y+15+j*28,f'd{j+1}' if j<2 else '…',18,v.BLUE)
        v.line(x+176,y+37,x+243,y+37,v.BLUE,1.6,arrow=True)
        v.rect(x+257,y+24,100,26,'#e0eee9',v.TEAL,2);v.text(x+368,y+45,'d',22,v.TEAL,600)
    def patches(x,y,size):
        for i,stem in enumerate(geo['views']):
            vi=next(a for a in m['views'] if a['stem']==stem)
            v.image(stem+'_crop.png',x+i*(size+10),y,size,size)
            origin=np.array(vi['crop'][:2]);observed=(np.array(vi['observed_xy'])-origin)/192
            u=x+i*(size+10)+observed[0]*size;w0=y+observed[1]*size
            v.line(u-5,w0,u+5,w0,'white',1.5);v.line(u,w0-5,u,w0+5,'white',1.5)
            for key,col in [('depth',v.BLUE),('ray',v.TEAL)]:
                p=(np.array(vi['predicted_xy'][key])-origin)/192
                v.circle(x+i*(size+10)+p[0]*size,y+p[1]*size,4,col,'none')
    def online(y):
        v.line(40,y-34,1760,y-34,v.LIGHT,1.5,'6 6');v.text(40,y,'ONLINE',21,v.TEAL,700)
        v.image('art_query.png',45,y+30,315,177)
        v.text(400,y+107,'Local features',24,v.INK,600);v.text(400,y+145,'Frozen SuperPoint',20,v.MUTED)
        v.line(695,y+112,760,y+112,width=2,arrow=True)
        v.text(800,y+107,'Sparse 2D–3D matching',24,v.INK,600);v.text(800,y+145,'Confidence filtering',20,v.MUTED)
        v.line(1185,y+112,1250,y+112,width=2,arrow=True)
        v.camera(1330,y+107,v.TEAL);v.text(1380,y+108,'Robust PnP → R, t',25,v.INK,600)
        v.text(1380,y+147,'One pose-solver call',20,v.MUTED)
        v.text(900,y+244,'No online scene rendering  ·  No additional dense refinement',23,v.TEAL,anchor='middle')
    # A: horizontal evidence-to-map layout, mechanisms connected by operations.
    begin(1090);v.text(40,45,'AnyGSLoc',31,v.INK,700);v.text(260,45,'Gaussian evidence → compact localization landmarks',27,v.INK)
    for x,n,title,col in [(40,'01','Render evidence',v.BLUE),(480,'02','Screen observations',v.BLUE),(920,'03','Rebuild landmarks',v.GOLD),(1360,'04','Select the map',v.TEAL)]:stage(x,106,n,title,col)
    v.image('art_gaussians.png',40,165,170,142);v.line(220,235,255,235,width=2,arrow=True);v.image(hero['stem']+'_crop.png',265,165,174,142)
    v.text(40,339,'Gaussian support',20,v.MUTED);v.text(269,339,'Rendered RGB',20,v.MUTED)
    v.image('art_depth.png',40,373,180,139);v.image('art_structure.png',253,373,180,139)
    v.text(40,545,'Native depth',20,v.MUTED);v.text(253,545,'Structure score',20,v.MUTED)
    v.image('art_validity.png',485,165,345,275);v.text(480,478,'Color: valid',23,v.TEAL,600);v.text(665,478,'Gray: excluded',23,v.MUTED,600)
    v.text(480,522,'Per-observation validity',22,v.INK);v.text(480,558,'before cross-view association',20,v.MUTED)
    patches(927,165,106);v.text(920,311,'Tracks define the landmark',22,v.INK,600)
    v.text(920,353,'Fixed-camera triangulation',21,v.INK);fusion(920,392)
    v.text(920,527,'Multi-view descriptor fusion',21,v.INK);v.text(920,563,'One X and one d per landmark',20,v.TEAL)
    v.image('art_map_full.png',1350,151,188,206);v.image('art_map_compact.png',1565,151,188,206)
    v.line(1536,246,1557,246,width=1.5,arrow=True)
    v.text(1450,390,f'{int(mp["full_count"]):,}',25,v.MUTED,600,'middle');v.text(1655,390,f'{int(mp["compact_count"]):,}',25,v.TEAL,600,'middle')
    v.text(1360,450,'Matching coverage',22,v.INK,600);v.text(1360,484,'+ pose-information diversity',20,v.INK)
    v.text(1360,548,'Deploy XYZ + descriptor',21,v.TEAL,600)
    for x in [448,885,1313]:v.line(x,290,x+24,290,width=2,arrow=True)
    v.text(40,625,'Illustrative data: StMarysChurch (01–03) and OldHospital (04); descriptor glyphs are schematic.',18,v.MUTED)
    online(720)
    v.add('<path d="M1620,576 L1620,661 L1060,661 L1060,792" fill="none" stroke="#187b70" stroke-width="2" marker-end="url(#arrow)"/>')
    v.text(1420,650,'compact map',18,v.TEAL,anchor='middle')
    finish('D02_overview_evidence_flow',1090)
    # B: central landmark construction, with input evidence and output compact map.
    begin(1110);v.text(40,45,'AnyGSLoc',31,v.INK,700);v.text(260,45,'Separate rendering primitives from localization landmarks',27,v.INK)
    stage(40,109,'a','Gaussian evidence',v.BLUE);stage(610,109,'b','Landmark construction',v.GOLD);stage(1320,109,'c','Compact selection',v.TEAL)
    v.image('art_gaussians.png',40,164,192,155);v.image(hero['stem']+'_crop.png',277,164,218,155)
    v.line(240,242,268,242,width=2,arrow=True)
    v.image('art_depth.png',40,356,190,149);v.image('art_validity.png',277,356,218,149)
    v.text(40,546,'Rendered depth',21,v.MUTED);v.text(277,546,'V2 validity overlay',20,v.TEAL)
    v.text(277,582,'Color: valid · gray: excluded',17,v.MUTED)
    patches(610,177,173)
    v.text(610,375,'Mutual matches → epipolar check → cycle / tracks',19,v.MUTED)
    v.text(610,409,'Observed correspondences determine X',24,v.INK,600)
    v.text(610,449,'Multi-ray geometry with fixed cameras',22,v.INK)
    fusion(646,477);v.text(610,601,'Track-level fusion determines d',24,v.INK,600)
    v.image('art_map_full.png',1300,159,202,220);v.image('art_map_compact.png',1530,159,202,220)
    v.line(1501,262,1525,262,width=2,arrow=True)
    v.text(1390,414,f'{int(mp["full_count"]):,}',25,v.MUTED,600,'middle');v.text(1630,414,f'{int(mp["compact_count"]):,}',25,v.TEAL,600,'middle')
    v.text(1320,470,'Coverage + pose information',22,v.INK,600);v.text(1320,514,'Complementary constraints',21,v.MUTED)
    v.text(1320,572,'Only the sparse map is deployed',21,v.TEAL,600)
    v.line(520,300,573,300,width=2,arrow=True);v.line(1180,300,1264,300,width=2,arrow=True)
    v.text(40,655,'StMarysChurch evidence and track; OldHospital maps. White + denotes an observation; blue/green show depth/ray projections.',17,v.MUTED)
    online(740)
    v.add('<path d="M1620,594 L1620,692 L1060,692 L1060,812" fill="none" stroke="#187b70" stroke-width="2" marker-end="url(#arrow)"/>')
    v.text(1420,682,'compact map',18,v.TEAL,anchor='middle')
    finish('D03_overview_landmark_centric',1110)
    (OUT/'layout_notes.json').write_text(json.dumps({'query':str(query),'status':'two review compositions, not final author-approved overview',
        'descriptor_glyphs':'schematic representation of aggregation across all valid track observations',
        'input_scenes':['StMarysChurch','OldHospital'],'map_point_size':'identical in full/compact views',
        'limitations':['The Gaussian panel is a sampled covariance projection, not the full alpha-composited prior.',
                       'Different-scene panels illustrate stages, not a single end-to-end anchor trajectory.']},indent=2))
    print('Two overview compositions exported',flush=True)

if __name__=='__main__':main()
