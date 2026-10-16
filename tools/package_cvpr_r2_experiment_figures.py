"""Package revision-two publication figures without changing experiment artifacts."""
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
OUT = ROOT / 'paper/experiments_cvpr_r2_20261003'
SPECS = [
    ('E1_final_localization', 'Final localization', 'build_cvpr_r2_e1_localization.py'),
    ('E2_geometry_source_evidence', 'Geometry consistency', 'build_cvpr_r2_e2_geometry.py'),
    ('E3_compact_map', 'Compact landmark maps', 'build_cvpr_r2_e3_compact.py'),
    ('E4_gaussian_prior', 'Gaussian-prior comparison', 'build_cvpr_r2_e4_prior.py'),
    ('E5_deployment', 'Accuracy versus deployment cost', 'build_cvpr_r2_e5_deployment.py'),
]
CAPTIONS = [
    r"\textbf{Qualitative final-pose comparison.} Real queries and aligned reference renders compare STDLoc, ULF-Loc, and AnyGSLoc with its compact map. Alternating horizontal bands use the ground-truth and recorded final poses; consistent structures across band boundaries indicate agreement. All methods in a row share one frozen reference renderer, camera calibration, crop, and display scale. Labels give translation and rotation errors. These examples were selected to illustrate AnyGSLoc's advantage using recorded pose errors and reference-point image displacement; they are not a representative performance estimate. Gaussian rendering is used only for offline visualization and is absent from AnyGSLoc deployment.",
    r"\textbf{Geometry consistency with fixed landmark identities and descriptors.} Empirical cumulative distributions of reprojection residuals on all 1,289,079 archived observations of 55,817 common landmarks in StMarysChurch. Only XYZ changes between center aggregation, depth aggregation, and multiview rays. All methods use the same evaluation observations and cameras; invalid projections remain in the denominator. Ray coordinates were estimated from these mapping observations, making this a mapping-fit diagnostic rather than held-out query accuracy. The error axis is logarithmic.",
    r"\textbf{Full and compact landmark maps.} Matched views and detail strips use identical cameras, crops, point sizes, and descriptor PCA colors. One PCA basis and color scale fitted to the Full maps are reused for Compact. A common visibility band derived from archived rendered depth is applied symmetrically to both maps for display. Whole-map counts include every stored landmark; detail counts refer to the displayed band. Across five scenes, selection reduces 535,840 landmarks to 176,329 while retaining recognizable facade structure in these views. The visualization demonstrates sparsity, without claiming geometric denoising.",
    r"\textbf{Different Gaussian priors, compact maps, and final poses.} Two real queries compare five frozen priors. Prior-render rows show appearance at the reference camera; map rows use a shared descriptor PCA basis and color scale. Pose-alignment rows alternate ground-truth and estimated-pose bands rendered from one common reference model per query, with translation and rotation errors. The examples were selected for comparable low localization errors among MAtCHA, 2DGS, 3DGS, and i3DGS+1k, while retaining the recorded i3DGS-raw outcomes. Rendering serves offline inspection only and is not part of AnyGSLoc localization. Prior preparation differs; these examples do not estimate average appearance quality.",
    r"\textbf{Final accuracy versus deployment cost.} Both panels share recall at 5\,cm/$5^\circ$ on the horizontal axis; costs use logarithmic vertical axes. (a) Total warmed query latency includes both coarse and refinement stages for STDLoc and ULF-Loc; AnyGSLoc uses one sparse localization stage. Recall and latency are scene macro means. (b) Native scene-specific inference assets summed over five scenes, excluding shared pretrained weights, query inputs, and intermediate training artifacts. Native configurations and resolutions differ. Timing excludes outer I/O and pose-error evaluation but retains ULF-Loc's internal reference-camera inlier diagnostic. Storage preserves original formats and overhead rather than assuming optimal compression.",
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
        assert dpi >= 300, (stem, dpi)
        review.insert_pdf(doc)
    review.set_toc([[1, f'E{i + 1} — {title}', i + 1]
                    for i, (_, title, _) in enumerate(SPECS)])
    review.set_metadata({'title': 'AnyGSLoc — experiment figures, revision 2',
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
    lines = ['REVISION 2 · Five publication figures', '',
             'E1 / E4: GT–estimate render alignment',
             'E2: paired mapping-residual distributions',
             'E3: matched Full / Compact views',
             'E5: accuracy on x, deployment cost on y', '',
             'PDF / SVG / high-resolution PNG',
             'Captions, source scripts, and provenance',
             'Existing experiments; offline visualization only']
    for j, line in enumerate(lines):
        draw.text((x + 30, y + 45 + j * 48), line, font=small, fill='#263341')
    canvas.save(OUT / 'CONTACT_SHEET.png')

    tex = ['% Requires graphicx; copy PDFs into your CVPR paper and adjust paths.',
           '% Figure insertion snippets, not a standalone LaTeX document.', '']
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
              'figure_cache_reader.py', 'package_cvpr_r2_experiment_figures.py']
    for script in scripts:
        shutil.copy2(ROOT / 'tools' / script, sources / script)
    (OUT / 'README.txt').write_text('''AnyGSLoc — CVPR 实验图第二轮，2026-10-03

先看 REVIEW_all_five.pdf（五页独立原尺寸图），CONTACT_SHEET.png 为导航总览。
每图有 PDF、SVG、高清 PNG。论文推荐插入 PDF，矢量文字和图表配合真实图像栅格。
E2 按单栏 3.45 英寸绘制，其余按双栏 7.1 英寸绘制。
figures.tex 是插入片段；CAPTIONS.txt 是简洁英文图注，各 E*.caption.txt 是详细证据说明。

本轮修改：
E1：优势样本、共同参考模型的 GT / final-pose 渲染横带拼接，移除 case 分类。
E2：StMarysChurch 全部共同轨迹观测的真实残差 ECDF，移除突兀 crop 与重复定位统计。
E3：移除 selection overlay，Full / Compact 两列、对应建筑细节与真实压缩幅度。
E4：两个 query、五种 prior 的外观 / PCA 地图 / 共同参考 pose 对齐，移除汇总表格。
E5：横轴精度、纵轴时延或存储；baseline 使用 coarse + refinement 总耗时。

解释与使用边界：
- 仅使用已有 query / pose / map / timing；新增工作为离线渲染、统计和排版。
- E1 / E4 的 Gaussian 只用于离线展示，不是 AnyGSLoc 在线依赖；同一例的 pose 面板
  使用同一参考模型。选例是有意展示优势和兼容性的定性例子，整体结论仍以完整测试集为准。
- E2 是 mapping-fit 诊断，ray 坐标曾使用这些观测估计；不能写成独立 query 泛化精度。
- E3 的相同 surface band 是可视化遮挡规则，不修改地图；PCA 是描述子色彩，不是语义。
  本图支持稀疏性和可见结构保留，不证明几何去噪或每个场景定位均改善。
- E4 两个视角的局部外观差异不能推导一种先验平均质量更好。
- E5 使用原生配置；不同分辨率/设置不能解释为严格控制条件下的纯推理加速比。
  ULF-Loc 的 native reference-camera inlier 诊断仍含在计时中，外部pose误差评估不含。
  Coarse prefix 包含共享 query 预计算；阶段拆分见 E5_cost_breakdown.csv。

存储为五个 Cambridge 场景原生推理资产之和（decimal MB）：
AnyGSLoc compact 184.095966；ULF-Loc final 526.907219；STDLoc final 2226.207728。
包含必要 Gaussian / descriptor / scene detector / index，排除共享权重、训练中间文件、
重复 benchmark 与 query 输入。保留原始格式开销；逐文件依据和哈希见 E5_storage_audit.json。

source 是出图脚本快照；assets 保留离线图像、选例和统计缓存，包括未最终采用的候选。
重新生成需要 /root/STDLoc 仓库模块与原始 /mnt/pool/sqy 实验产物；并未复制大型原始数据。
环境 /root/miniconda3/envs/g4splat/bin/python；E1 / E4 的 GPU 渲染可使用现有缓存跳过。
重画：tools/build_cvpr_r2_e1_localization.py draw --style bands；
tools/build_cvpr_r2_e2_geometry.py --redraw；tools/build_cvpr_r2_e3_compact.py --redraw；
tools/build_cvpr_r2_e4_prior.py draw；tools/build_cvpr_r2_e5_deployment.py；
最后运行 tools/package_cvpr_r2_experiment_figures.py。
R2 E5 读取上一轮经审计的数据；E3重新准备显示缓存也读取上一轮PCA产物，上一轮目录必须保留。

print_checks.json 记录每图尺寸、字体嵌入、越界检查、PNG分辨率；manifest.json 为文件哈希。
第一轮产物保持在相邻 experiments_cvpr_20261003 目录。
''')
    files = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    manifest = {'date': '2026-10-03', 'revision': 2, 'status': 'five_figure_review_set',
                'no_new_algorithm_experiments': True, 'figures': [s[0] for s in SPECS],
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
