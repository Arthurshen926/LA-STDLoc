"""Assemble the five scientific figures, publication captions and print checks."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

sys.path.insert(0, '/root/.cache/anygsloc_figure_deps')
import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/experiments_cvpr_20261003'
SPECS = [
    ('E1_final_localization', 'Final localization', 'build_cvpr_e1_localization.py'),
    ('E2_geometry_source_evidence', 'Geometry-source evidence', 'build_cvpr_e2_geometry.py'),
    ('E3_compact_map', 'Compact landmark maps', 'build_cvpr_e3_compact.py'),
    ('E4_gaussian_prior', 'Gaussian-prior sensitivity', 'build_cvpr_e4_prior.py'),
    ('E5_deployment', 'Accuracy versus deployment cost', 'build_cvpr_e5_deployment.py'),
]
CAPTIONS = [
    r"\textbf{Final localization on real queries.} Green circles and magenta crosses project identical reference-cloud landmarks under the ground-truth and recorded final poses, respectively. Every method uses the same RGB image and $4\times$ crop in each row; labels give TE/RE. From 1,918 common valid queries, the typical case is nearest the pooled AnyGSLoc median within the common-success set ($<15$\,cm, $<5^\circ$); favorable and adverse cases are nearest the symmetric 2nd/98th percentiles of the error gap to the better baseline final pose. The geometry consists of COLMAP reference points, not extracted surface edges. AnyGSLoc uses its compact map.",
    r"\textbf{Changing landmark coordinates with fixed identities and descriptors.} (a) Three recorded mapping patches and an independent real query share one KingsCollege landmark. Crosses mark recorded image features; common pixel axes show reprojection offsets under reference cameras. (b,c) Test translation error and recall across all five scenes, with KingsCollege retained as a counterexample. Center/depth aggregate the same usable observations; ray coordinates retain their original track support. The local correspondence illustrates the coordinate estimators and is not an average query statistic. GC/KC/OH/SF/SM denote GreatCourt, KingsCollege, OldHospital, ShopFacade, and StMarysChurch.",
    r"\textbf{What compact selection removes and retains.} Each row uses the same camera, crop, point size and descriptor colors for Full and Compact. One PCA fitted to the displayed Full maps and fixed percentile bounds color all panels; colors describe SuperPoint descriptors, not semantics. Overlays show removed landmarks in gray and retained landmarks in color. Insets enlarge an identical dense region and report its actual retention. OldHospital improves in test accuracy while KingsCollege degrades. Scene counts above the rows and local inset counts have distinct scopes; the bottom totals cover all five Cambridge scenes: 535,840 to 176,329 landmarks and 559.43 to 184.10\,MB.",
    r"\textbf{Gaussian priors, compact maps, and localization.} (a) Offline renders of five frozen priors share one camera, resolution and display mapping. (b) Compact landmarks use one PCA fitted to equal-size descriptor samples from every prior and common color scaling; $N$ counts the scene map. (c) An identical real-query crop shows common COLMAP-point projections under reference and estimated poses, with TE/RE. The query is nearest the median average normalized-error rank across all five priors in StMarysChurch. (d) Five-scene macro results. Upstream preparation differs across priors; these results characterize compatibility and sensitivity. Offline rendering is absent from AnyGSLoc localization.",
    r"\textbf{Accuracy versus online deployment cost.} (a) Recall at 5\,cm/$5^\circ$ versus warmed mean pipeline latency. Arrows connect native coarse prefixes, including shared query precomputation, to cumulative final stages. (b) Final accuracy versus native scene-specific inference assets summed across five scenes, excluding shared weights, training artifacts and query inputs. Labels give means of scene-median TE/RE; recall and latency are scene macro means. Native configurations and resolutions differ. Timing excludes outer I/O and pose-error evaluation, but retains ULF-Loc's internal reference-camera inlier diagnostic. Storage follows verified source dependencies without repacking and is not an optimal-compression lower bound. AnyGSLoc uses its compact map.",
]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8 << 20),b''):h.update(b)
    return h.hexdigest()

def main():
    review = fitz.open()
    checks = []
    for stem,title,script in SPECS:
        for ext in ['pdf','png','svg']:
            assert (OUT/f'{stem}.{ext}').is_file(), (stem,ext)
        doc=fitz.open(OUT/f'{stem}.pdf');assert len(doc)==1
        page=doc[0]
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
        offpage=[s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox'])+(-.5,-.5,.5,.5))]
        fonts=[]
        for item in page.get_fonts():
            xref=item[0]; meta=doc.extract_font(xref)
            fonts.append({'name':item[3],'subtype':item[2],'embedded':bool(meta[3])})
        png=Image.open(OUT/f'{stem}.png')
        checks.append({'figure':stem,'pdf_pages':len(doc),'pdf_inches':[page.rect.width/72,page.rect.height/72],
            'png_pixels':list(png.size),'png_nominal_dpi':png.width/(page.rect.width/72),
            'min_pdf_font_pt':min(s['size'] for s in spans),'off_page_text':offpage,'fonts':fonts})
        assert not offpage,(stem,offpage)
        assert all(f['embedded'] for f in fonts),(stem,fonts)
        assert not any(f['subtype']=='Type3' for f in fonts),(stem,fonts)
        assert png.width >= 2100
        review.insert_pdf(doc)
    review.set_toc([[1, f'E{i+1} — {title}', i+1] for i,(_,title,_) in enumerate(SPECS)])
    review.set_metadata({'title':'AnyGSLoc — five experiment figures','author':'AnyGSLoc figure suite',
        'subject':'CVPR-style review set from existing experimental artifacts'})
    review.save(OUT/'REVIEW_all_five.pdf',garbage=4,deflate=True)
    (OUT/'print_checks.json').write_text(json.dumps(checks,indent=2))

    # Contact sheet is a navigation aid; individual PDFs retain their native size.
    canvas=Image.new('RGB',(1800,2000),'#f3f5f7');draw=ImageDraw.Draw(canvas)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,22);small=ImageFont.truetype(fontpath,20)
    for i,(stem,title,_) in enumerate(SPECS):
        x=24+(i%2)*888;y=24+(i//2)*650
        draw.rounded_rectangle((x,y,x+864,y+628),radius=8,fill='white')
        draw.text((x+18,y+16),f'E{i+1}  {title}',font=font,fill='#243349')
        im=Image.open(OUT/f'{stem}.png').convert('RGB');im.thumbnail((842,555),Image.Resampling.LANCZOS)
        canvas.paste(im,(x+(864-im.width)//2,y+58+(555-im.height)//2))
    x,y=912,1324
    draw.rounded_rectangle((x,y,x+864,y+628),radius=8,fill='white')
    lines=['Five figures · Existing experimental artifacts','',
        'Vector text / plots in PDF and SVG','High-resolution PNG previews',
        'English captions and LaTeX insertion snippets','Scripts, source hashes, and storage audit','',
        'Five-scene native representation storage:',
        'AnyGSLoc compact     184.1 MB','ULF-Loc final             526.9 MB','STDLoc final            2,226.2 MB']
    for j,line in enumerate(lines):draw.text((x+28,y+35+j*43),line,font=small,fill='#243349')
    canvas.save(OUT/'CONTACT_SHEET.png')

    tex=['% Add graphicx to the paper preamble. Copy selected figures into your CVPR project.',
         '% Paths below are relative to this figure directory; adjust for your paper layout.',
         '% These are insertion snippets, not a standalone LaTeX document.','']
    for (stem,title,_),caption in zip(SPECS,CAPTIONS):
        tex += [r'\begin{figure*}[t]',r'  \centering',
                rf'  \includegraphics[width=\textwidth]{{{stem}.pdf}}',
                '  \\caption{'+caption+'}',rf'  \label{{fig:{stem.lower()}}}',r'\end{figure*}','']
    (OUT/'figures.tex').write_text('\n'.join(tex))
    (OUT/'CAPTIONS.txt').write_text('\n\n'.join(title+'\n'+caption for (_,title,_),caption in zip(SPECS,CAPTIONS))+'\n')
    sources=OUT/'source';sources.mkdir(exist_ok=True)
    for _,_,script in SPECS:
        shutil.copy2(ROOT/'tools'/script,sources/script)
    for script in ['package_cvpr_experiment_figures.py','figure_cache_reader.py']:
        shutil.copy2(ROOT/'tools'/script,sources/script)
    readme='''AnyGSLoc — CVPR 实验结果图，2026-10-03

先看 REVIEW_all_five.pdf（每页一张），CONTACT_SHEET.png 为总览。
E1 Final localization：三种方法的真实 query 与中立参考点投影，典型/优势/不利案例。
E2 Geometry source：相同 identity/descriptor 下的 XYZ 对照、独立 query 残差、五场景结果。
E3 Compact map：共享 descriptor PCA，Full → selection → Compact，局部放大与真实留存率。
E4 Gaussian prior：五种 prior 的同 pose 渲染 → 共享 PCA 地图 → 同 query 定位与汇总。
E5 Deployment：精度对时延、精度对横向部署存储；原生资产依赖审计已补全。

每张提供 PDF、SVG、高清 PNG。建议论文插入 PDF；文字和图表线条为矢量，真实图像和密集点云为栅格。
figures.tex 是 figure* 插入片段；CAPTIONS.txt 为精简英文论文图注。各 E*.caption.txt / *_caption.txt
保留详细证据边界。E*.json、assets 与 source 保留选例、PCA、统计和重现信息。
E3 可作为正文候选或补充材料；本目录不改动现有论文工程与旧图件。

存储（五个 Cambridge 场景总和；decimal MB）：
  AnyGSLoc compact   184.095966 MB
  ULF-Loc final      526.907219 MB
  STDLoc final      2226.207728 MB
统计原生最终推理表示、采样索引、必要描述子和场景 detector 等，剔除共享权重、训练日志、
中间 checkpoint、重复 benchmark 文件和 query 输入。保留原始格式与开销，并非理论最小压缩容量。
完整逐文件哈希及加载依据见 E5_storage_audit.json；两种 baseline mapping-mask 设置逐景值见CSV。

必要解释：
- 实验精度、pose、地图来自已有产物；新增的是可视化、离线 render 和统计，没有新算法实验。
- E1/E4 的绿色/洋红叠加是同一组 COLMAP 点的投影，不能称为结构边缘或可见表面真值。
- E3/E4 的 PCA 色彩表示描述子空间，不是语义；同一图内固定 PCA 和颜色缩放。
- E2 的 identities/descriptors 已逐场景核验相同；ray estimator保留原有track support。
- E5 是不同原生配置下的既有 warmed pipeline 计时。Coarse包含共享预计算；
  ULF-Loc内部GT inlier诊断仍计时，外部pose-error evaluation不计；不能视为等分辨率纯推理加速比。

重现（在 /root/STDLoc 仓库，需保留原始 /mnt/pool/sqy 实验产物）：
  /root/miniconda3/envs/g4splat/bin/python tools/build_cvpr_e1_localization.py
  /root/miniconda3/envs/g4splat/bin/python tools/build_cvpr_e2_geometry.py
  /root/miniconda3/envs/g4splat/bin/python tools/build_cvpr_e3_compact.py
  /root/miniconda3/envs/g4splat/bin/python tools/build_cvpr_e4_prior.py draw
  /root/miniconda3/envs/g4splat/bin/python tools/build_cvpr_e5_deployment.py --plot-only
  /root/miniconda3/envs/g4splat/bin/python tools/package_cvpr_experiment_figures.py
E4 的离线 render 缓存已随 assets 保存；无需重复 GPU render。
重新审计 E5 时去掉 --plot-only，会校验和hash已有native资产；不会启动定位实验。

检查：print_checks.json 记录每图尺寸、字体嵌入和PNG尺寸；各图已按双栏尺寸人工检查。
源代码快照在 source；重现需现有仓库模块和原始大型数据，不是单独打包全部实验数据。
'''
    (OUT/'README.txt').write_text(readme)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json']
    manifest={'date':'2026-10-03','status':'five_figure_review_set','figures':[s[0] for s in SPECS],
        'no_new_algorithm_experiments':True,'file_sha256':{str(p.relative_to(OUT)):sha(p) for p in files}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)))
    print(json.dumps({'zip':str(archive),'zip_MB':archive.stat().st_size/1e6,'checks':checks},indent=2))

if __name__=='__main__':main()
