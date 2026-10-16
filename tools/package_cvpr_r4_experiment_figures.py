"""Package revision-four publication figures without changing experiment artifacts."""
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
OUT = ROOT / 'paper/experiments_cvpr_r4_20261003'
SPECS = [
    ('E1_final_localization', 'Coarse-to-fine localization', 'build_cvpr_r4_e1_localization.py'),
    ('E2_geometry_source_evidence', 'Geometry consistency', 'build_cvpr_r4_e2_geometry.py'),
    ('E3_compact_map', 'Compact landmark maps', 'build_cvpr_r4_e3_compact.py'),
    ('E4_gaussian_prior', 'Gaussian-prior comparison', 'build_cvpr_r4_e4_prior.py'),
    ('E5_deployment', 'Accuracy versus deployment cost', 'build_cvpr_r4_e5_deployment.py'),
]
CAPTIONS = [
    r"\textbf{Coarse and final pose alignment.} STDLoc and ULF-Loc show their archived coarse and refined final poses; AnyGSLoc uses one compact-map localization stage. Bands 1 and 3 are renders at the ground-truth pose; bands 2 and 4 are renders at the estimated pose. Green and magenta identify these sources, not correct and incorrect regions. All methods share the reference Gaussian, intrinsics, crop, scale, and band pattern within a query. Labels give translation and rotation errors. These deliberately selected favorable examples illustrate alignment; aggregate performance is reported separately. The reference model is used only for offline visualization.",
    r"\textbf{Reprojection consistency of coordinate sources.} Empirical cumulative distributions of native-pixel reprojection errors on 1,289,079 paired mapping observations of 55,817 common landmarks across 1,465 contributing views in St. Mary's Church. Landmark identities and descriptors are fixed. All three XYZ sources use the same evaluation observations and cameras; invalid projections remain in the denominator, including beyond the displayed range. Ray coordinates were estimated from these mapping observations, so this measures mapping consistency rather than held-out query accuracy. The error axis is logarithmic.",
    r"\textbf{Full and compact landmark maps.} Matched views and details use the same camera, crop, point size, and visibility band derived from archived depth. Descriptor colors use one PCA basis and normalization fitted to the Full maps and reused for Compact. Whole-map counts cover every stored landmark; detail counts cover the displayed band. Across five scenes, selection reduces 535,840 landmarks to 176,329. The map-selection ablation reports scene-dependent localization trade-offs, including a recall decrease in King's College and an increase in Old Hospital. Recognizable structure in these views does not imply lossless localization or geometric denoising.",
    r"\textbf{Gaussian priors, landmarks, and pose alignment.} Prior-render rows show each input prior at the same GT camera; compact-landmark rows use one shared descriptor PCA and normalization fitted to equal samples from all ten scene/prior maps. $N$ is the total scene-map landmark count. Pose-alignment rows use one common reference model per query, alternating renders at the GT and estimated poses; the colored edges identify sources. Thus band discontinuities expose pose disagreement within a shared renderer, not differences between input-prior renderers. These selected examples have similar low errors for the four non-raw priors and larger raw-i3DGS errors. Rendering is offline only; prior preparation differs.",
    r"\textbf{Stage-wise accuracy versus deployment cost.} (a) Mean pipeline latency to each displayed stage. Hollow coarse and filled final points are joined by arrows labeled with the additional fine-stage time. Final latency includes the coarse prefix; AnyGSLoc uses one sparse stage. Recall and latency are scene-equal means. (b) Required scene-specific inference assets summed over five scenes, using a tighter recall range. Cost axes are logarithmic. Native configurations and resolutions differ. Timing excludes outer I/O and pose-error evaluation but retains ULF-Loc's reference-camera diagnostic. Storage includes necessary scene models, descriptors, detectors, and indices, preserves native formats, and excludes shared weights, query inputs, and intermediate training artifacts.",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    review = fitz.open()
    checks = []
    for stem, title, script in SPECS:
        for ext in ['pdf', 'png', 'svg', 'caption.txt', 'provenance.json']:
            assert (OUT / f'{stem}.{ext}').is_file(), (stem, ext)
        doc = fitz.open(OUT / f'{stem}.pdf')
        assert len(doc) == 1
        page = doc[0]
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                 for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        offpage = [s['text'] for s in spans
                   if not page.rect.contains(fitz.Rect(s['bbox']) + (-.5, -.5, .5, .5))]
        fonts = []
        for item in page.get_fonts():
            meta = doc.extract_font(item[0])
            fonts.append({'name': item[3], 'subtype': item[2], 'embedded': bool(meta[3])})
        png = Image.open(OUT / f'{stem}.png')
        dpi = png.width / (page.rect.width / 72)
        checks.append({'figure': stem, 'pdf_pages': len(doc),
                       'intended_layout': 'single column' if stem.startswith('E2') else 'two columns',
                       'pdf_inches': [page.rect.width / 72, page.rect.height / 72],
                       'png_pixels': list(png.size), 'png_nominal_dpi': dpi,
                       'min_pdf_font_pt': min(s['size'] for s in spans),
                       'off_page_text': offpage, 'fonts': fonts})
        assert not offpage, (stem, offpage)
        assert all(f['embedded'] and f['subtype'] != 'Type3' for f in fonts), (stem, fonts)
        assert min(s['size'] for s in spans) >= 6.99, (stem, 'font smaller than 7 pt')
        assert dpi >= 300, (stem, dpi)
        review.insert_pdf(doc)
    review.set_toc([[1, f'E{i + 1} — {title}', i + 1]
                    for i, (_, title, _) in enumerate(SPECS)])
    review.set_metadata({'title': 'AnyGSLoc — experiment figures, revision 4',
                         'author': 'AnyGSLoc figure suite',
                         'subject': 'Existing experiments; revised scientific visualization'})
    review.save(OUT / 'REVIEW_all_five.pdf', garbage=4, deflate=True)
    (OUT / 'print_checks.json').write_text(json.dumps(checks, indent=2))

    canvas = Image.new('RGB', (1800, 2160), '#edf0f3')
    draw = ImageDraw.Draw(canvas)
    fontpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(fontpath, 23)
    small = ImageFont.truetype(fontpath, 21)
    for i, (stem, title, _) in enumerate(SPECS):
        x, y = 24 + (i % 2) * 888, 24 + (i // 2) * 708
        draw.rounded_rectangle((x, y, x + 864, y + 684), radius=8, fill='white')
        draw.text((x + 20, y + 17), f'E{i + 1}  {title}', font=font, fill='#263341')
        im = Image.open(OUT / f'{stem}.png').convert('RGB')
        im.thumbnail((842, 610), Image.Resampling.LANCZOS)
        canvas.paste(im, (x + (864 - im.width) // 2, y + 62 + (610 - im.height) // 2))
    x, y = 912, 1440
    draw.rounded_rectangle((x, y, x + 864, y + 684), radius=8, fill='white')
    lines = ['REVISION 4 · Five publication figures', '',
             'E1 / E4: explicit four-band source legend',
             'E2: concise scene title; unchanged residuals',
             'E3: shared PCA and compression trade-offs',
             'E5: latency and storage without FPS labels', '',
             'PDF / SVG / high-resolution PNG',
             'Captions, source scripts, and provenance',
             'Same data and cached renders; refined layout']
    for j, line in enumerate(lines):
        draw.text((x + 30, y + 45 + j * 48), line, font=small, fill='#263341')
    canvas.save(OUT / 'CONTACT_SHEET.png')

    tex = ['% Requires graphicx; copy PDFs into your CVPR paper and adjust paths.',
           '% Figure insertion snippets, not a standalone LaTeX document.',
           '% Existing table labels: tab:selection (macro) and tab:selection-scenes (archive).',
           '% Add a table cross-reference to E3 when the corresponding table is included in your paper.', '']
    for (stem, _, _), caption in zip(SPECS, CAPTIONS):
        env = 'figure' if stem.startswith('E2') else 'figure*'
        tex += [rf'\begin{{{env}}}[t]', r'  \centering',
                rf'  \includegraphics[width=\linewidth]{{{stem}.pdf}}',
                '  \\caption{' + caption + '}', rf'  \label{{fig:{stem.lower()}}}',
                rf'\end{{{env}}}', '']
    (OUT / 'figures.tex').write_text('\n'.join(tex))
    (OUT / 'CAPTIONS.txt').write_text('\n\n'.join(title + '\n' + caption
                                    for (_, title, _), caption in zip(SPECS, CAPTIONS)) + '\n')
    sources = OUT / 'source'
    sources.mkdir(exist_ok=True)
    scripts = [s[2] for s in SPECS] + ['build_cvpr_e1_localization.py', 'build_cvpr_e4_prior.py',
              'figure_cache_reader.py', 'build_cvpr_r2_e1_localization.py', 'build_cvpr_r2_e2_geometry.py',
              'build_cvpr_r2_e3_compact.py', 'build_cvpr_r2_e4_prior.py', 'build_cvpr_r3_e1_localization.py', 'build_cvpr_r3_e2_geometry.py',
              'build_cvpr_r3_e3_compact.py', 'build_cvpr_r3_e4_prior.py', 'package_cvpr_r4_experiment_figures.py']
    for script in scripts:
        shutil.copy2(ROOT / 'tools' / script, sources / script)
    (OUT / 'README.txt').write_text("AnyGSLoc — CVPR 实验图第四轮，2026-10-03\n\n先看 REVIEW_all_five.pdf；CONTACT_SHEET.png 为总览。\nPDF/SVG/高清PNG齐全。论文推荐PDF：文字/曲线是矢量，照片/点云是原始缓存图像。\nE2采用3.45英寸单栏，其余7.1英寸双栏。figures.tex为插入片段，不是独立LaTeX工程。\nCAPTIONS.txt为精简英文图注；各图caption/provenance和VISUALIZATION_PROTOCOL.txt记录解释与来源。\n\n本轮只调整展示和图注，沿用R3样本、既有pose、渲染、地图、误差和计时。\n无新增定位/训练/场景渲染，也未重做存储统计。前三轮目录完整保留。\n\nE1：保留3×6 coarse/fine/single布局；增加四条带来源示意，明确Render at GT pose与\nRender at estimated pose。绿色/洋红是图像来源，不是正确/错误区域。统一场景阅读名。\nE2：标题缩为St. Mary's Church，观测数量保留在caption。CDF数值和坐标轴不变。\nE3：补Shared descriptor PCA，统一场景名，减轻底部总数的强调。Full/Compact仍用相同PCA、\n点大小、相机、显示mask和ROI。caption连接选择消融，不在图内重复添加定位统计。\nE4：明确Prior render / Compact landmarks / Pose alignment，合并为全局四带说明。\n共用显示模型仍只用于底部位姿可视化，顶部依旧来自各自prior。保留误差单位与N外置标签。\nE5：移除图面FPS与换算脚注，保留两阶段箭头、fine增量和final总耗时。\n轴名为Mean pipeline latency，右图标题Deployment storage。FPS数值仍在源CSV，未删除审计记录。\n\n选择消融的真实对应文件：\npaper/AnyGSLoc_TRO_v6/tables/selection.tex：tab:selection。\npaper/AnyGSLoc_TRO_v6/tables/selection_scenes.tex：tab:selection-scenes（archive）。\n当前figure插入片段以自然语言连接该消融，未默认插入可能在CVPR工程中不存在的table引用。\n表中全局R5由43.00%到42.08%，点数由535,840到176,329；单场景有取舍。\n已核验Old Hospital R5 34.07%→35.71%，King's College 16.91%→12.83%。\n图的结构保留不能推导出定位无损、几何去噪或每个场景准确率提高。\n\n存储口径（五场景合计，decimal MB）保持：\nAnyGSLoc compact184.095966；ULF-Loc final526.907219；STDLoc final2226.207728。\n完整逐文件加载依据见E5_storage_audit.json，阶段时间见E5_cost_breakdown.csv。\nNative配置/分辨率存在差异；原生ULF-Loc reference-camera diagnostic仍含在pipeline计时里。\n外层I/O和pose-error evaluation排除；coarse prefix包括共享query预计算。\n\nsource为脚本快照。重现需要当前仓库模块、R1/R2/R3缓存与/mnt/pool/sqy原始产物。\npublication ZIP包含图件、图注、脚本、provenance及审计；完整ZIP另含本轮assets，未复制全部实验数据。\n主要环境：/root/miniconda3/envs/g4splat/bin/python。\n各工具脚本说明及CLI见source；最终打包脚本tools/package_cvpr_r4_experiment_figures.py。\nprint_checks.json记录尺寸、字体嵌入、边界及PNG分辨率；manifest.json记录各文件SHA256。\n")
    files = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    manifest = {'date': '2026-10-03', 'revision': 4, 'status': 'five_figure_review_set',
                'no_new_algorithm_experiments': True, 'no_new_scene_rendering': True, 'figures': [s[0] for s in SPECS],
                'file_sha256': {str(p.relative_to(OUT)): sha(p) for p in files}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(OUT.name) / p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    publication = OUT.parent / (OUT.name + '_publication.zip')
    publication_files = [p for p in files if 'assets' not in p.relative_to(OUT).parts]
    publication_manifest = dict(manifest)
    publication_manifest['status'] = 'publication_files_without_intermediate_assets'
    publication_manifest['full_archive'] = archive.name
    publication_manifest['file_sha256'] = {
        str(p.relative_to(OUT)): sha(p) for p in publication_files}
    with zipfile.ZipFile(publication, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in publication_files:
            z.write(p, str(Path(OUT.name) / p.relative_to(OUT)))
        z.writestr(str(Path(OUT.name) / 'manifest.json'), json.dumps(publication_manifest, indent=2))
        z.writestr(str(Path(OUT.name) / 'ARCHIVE_CONTENTS.txt'),
                   'Publication package: figures, captions, source scripts, provenance and storage audit.\n'
                   'Intermediate assets and offline render/statistics caches remain in the full archive: '
                   + archive.name + '\n')
    with zipfile.ZipFile(publication) as z:
        assert z.testzip() is None
    print(json.dumps({'zip': str(archive), 'zip_MB': archive.stat().st_size / 1e6,
                      'publication_zip': str(publication),
                      'publication_zip_MB': publication.stat().st_size / 1e6,
                      'hashed_files': len(files), 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
