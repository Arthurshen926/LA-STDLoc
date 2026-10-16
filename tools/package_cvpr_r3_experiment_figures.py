"""Package revision-three publication figures without changing experiment artifacts."""
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
OUT = ROOT / 'paper/experiments_cvpr_r3_20261003'
SPECS = [
    ('E1_final_localization', 'Coarse-to-fine localization', 'build_cvpr_r3_e1_localization.py'),
    ('E2_geometry_source_evidence', 'Geometry consistency', 'build_cvpr_r3_e2_geometry.py'),
    ('E3_compact_map', 'Compact landmark maps', 'build_cvpr_r3_e3_compact.py'),
    ('E4_gaussian_prior', 'Gaussian-prior comparison', 'build_cvpr_r3_e4_prior.py'),
    ('E5_deployment', 'Accuracy versus deployment cost', 'build_cvpr_r3_e5_deployment.py'),
]
CAPTIONS = [
    r"\textbf{Coarse and final pose alignment.} STDLoc and ULF-Loc each show their archived coarse and refined final poses; AnyGSLoc uses one compact-map localization stage. Each panel alternates strips rendered at the ground-truth and estimated poses with one shared reference Gaussian, calibration, crop, and scale per query. Labels give recorded translation and rotation errors. The same favorable examples are used at every stage; refinement is not assumed to improve every query. Rendering is an offline visualization tool and does not participate in AnyGSLoc localization.",
    r"\textbf{Geometry consistency with fixed landmark identities and descriptors.} Empirical cumulative distributions of native-pixel reprojection errors on all 1,289,079 common mapping observations of 55,817 landmarks across 1,465 contributing views in StMarysChurch. Only XYZ changes between center aggregation, depth aggregation, and multiview rays. All sources use the same evaluation observations and cameras; invalid projections remain in the denominator. Ray coordinates were estimated from these mapping observations, so the curves describe mapping consistency rather than held-out query accuracy. The error axis is logarithmic.",
    r"\textbf{Full and compact landmark maps.} Matched views and detail strips use identical cameras, crops, point sizes, and descriptor PCA colors. One PCA basis and color scale fitted to the Full maps are reused for Compact. A common visibility band derived from archived rendered depth is applied symmetrically for display. Whole-map counts include every stored landmark; detail counts refer to the displayed band. Across five scenes, selection reduces 535,840 landmarks to 176,329 while retaining recognizable facade structure in these views. The visualization demonstrates sparsity, without claiming geometric denoising.",
    r"\textbf{Gaussian priors and localization compatibility.} Each query compares prior appearance, shared-PCA compact landmarks, and final pose alignment. Alternating ground-truth and estimated-pose strips use a common reference Gaussian and identical structural zoom for every prior. The selected queries have comparable low errors for MAtCHA, 2DGS, 3DGS, and i3DGS+1k, while raw i3DGS causes visible window-frame or door-frame discontinuities. Labels give recorded translation and rotation errors. These are illustrative examples, not representative error samples; no pose displacement is exaggerated. Rendering is offline only, and prior preparation differs.",
    r"\textbf{Stage-wise accuracy versus deployment cost.} (a) Hollow coarse and filled final points are connected by arrows labeled with the additional fine-stage time. Final cost includes both stages; AnyGSLoc uses a single compact-map localization stage. Equivalent FPS is $1000/\bar t_{\mathrm{ms}}$, with $\bar t$ the scene-macro mean latency, not measured end-to-end throughput. (b) Final recall versus native scene-specific inference assets summed over five scenes, using a tighter recall range. Both cost axes are logarithmic. Native configurations and resolutions differ. Timing excludes outer I/O and pose-error evaluation but retains ULF-Loc's reference-camera inlier diagnostic. Storage preserves original formats and excludes shared weights, query inputs, and intermediate training artifacts.",
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
    review.set_metadata({'title': 'AnyGSLoc — experiment figures, revision 3',
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
    lines = ['REVISION 3 · Five publication figures', '',
             'E1: coarse / fine / single-stage alignment',
             'E2: paired mapping-residual distributions',
             'E4: visible raw-prior alignment differences',
             'E5: cumulative cost and equivalent FPS', '',
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
              'figure_cache_reader.py', 'build_cvpr_r2_e1_localization.py', 'build_cvpr_r2_e2_geometry.py',
              'build_cvpr_r2_e3_compact.py', 'build_cvpr_r2_e4_prior.py', 'package_cvpr_r3_experiment_figures.py']
    for script in scripts:
        shutil.copy2(ROOT / 'tools' / script, sources / script)
    (OUT / 'README.txt').write_text('AnyGSLoc — CVPR 实验图第三轮，2026-10-03\n\n先看 REVIEW_all_five.pdf；CONTACT_SHEET.png 为总览。\n每张提供 PDF/SVG/PNG。PDF 的文字/曲线为矢量；图像是真实归档数据或离线渲染。\nE2 为 3.45 英寸单栏，其余 7.1 英寸双栏。figures.tex 为插入片段，不是独立 LaTeX 工程。\nCAPTIONS.txt 为论文精简图注；单图 caption/provenance 保留详细定义和证据范围。\n\n本轮修改：\nE1：3×6布局，STDLoc/ULF-Loc 各展示 coarse → fine(final)，AnyGSLoc single。\n保持上一轮三个样本与相同 ROI，补充既有 coarse pose 的离线渲染。ShopFacade 的 ULF-Loc\nrefinement略差于coarse，如实保留；不是通过挑选强制制造每次fine都会变好的印象。\nE2：曲线不变。残差是二维像素重投影误差，原生相机/feature grid中计算。\n修正了一个元数据计数：1,465个视图实际贡献共同观测；完整归档有1,487个视图，22个未贡献。\nE3：保留R2 Full/Compact共享PCA、对应细节与压缩幅度，图件字节不变。\nE4：重选两个raw有实际可见偏移、其余四prior仍定位接近的query；统一更紧结构ROI与高分辨率\n参考渲染，窗棂/门框真实错位明显。所有prior使用相同crop/缩放，未夸大位移。\nE5：两阶段两个端点，连线上标fine增量，final点是coarse+fine总耗时；补充等效FPS。\n\n关于三个时间量和FPS：\nSTDLoc：261.07 ms coarse + 142.35 ms fine = 403.42 ms total，等效2.48 FPS。\nULF-Loc：570.75 + 798.82 = 1369.57 ms total，等效0.73 FPS。\nAnyGSLoc：67.10 ms single，等效14.90 FPS。\nFine是增量成本，不是独立于coarse的第三个位姿/精度结果，所以不画第三个精度端点。\n等效FPS是1000/scene-macro mean latency，不是完整I/O与外围循环的实测吞吐率，也不是各scene\nFPS的均值。图注保留原生配置/分辨率差异，ULF-Loc native reference-camera diagnostic仍在计时中。\nR3的runtime子图覆盖coarse与final精度，storage子图仅覆盖final范围，横轴范围因此不同且明示。\n\n重投影误差：\ne_ij = || pi(K_i (R_i X_j+t_i)) - (0.5,0.5) - u_ij ||_2。\n半像素项匹配原归档feature-grid约定。三种XYZ中位残差12.36 / 2.42 / 0.74 px。\n这是mapping-fit诊断：ray曾使用同一批建图观测估计。可解释几何一致性，不能当作独立query\n泛化精度或绝对三维精度；定位结果仍由单独实验表支持。详见E2_independent_audit.json。\n\n展示方法的论文先例：\nSTDLoc Fig.8和补充Fig.D采用query/render拼接；GS-CPR Fig.4采用GT image/estimated render\n对角线拼接。STDLoc官方页面说明ACE/HLoc不具备NVS，仍共用Feature Gaussian可视化。\n没有查到完全相同的GT-render/estimated-render交替横带先例，因此只定义为本图展示方式。\n已核验原始论文图号/页面与URL，详见 ALIGNMENT_REFERENCES.txt。\n\n五场景存储保持原已审计数据（decimal MB）：\nAnyGSLoc compact 184.095966；ULF-Loc final 526.907219；STDLoc final 2226.207728。\n逐文件加载依据和hash见 E5_storage_audit.json；阶段时间与FPS见 E5_cost_breakdown.csv。\n\n只用已有query/pose/map/timing；新增工作是离线render、统计审计和排版，无新定位/训练实验。\nPCA为descriptor coloring，不是semantic labels。E3不声称去噪或所有scene定位均提升。\nE1/E4优先可解释的优势/兼容性案例，并非随机或代表性采样。共用Gaussian仅用于offline展示。\n\n文件组织与重现：\nsource为R3脚本与相关helper快照；原仓库模块、R1/R2目录及/mnt/pool/sqy原始产物必须保留。\n完整ZIP含R3新增assets；publication ZIP仅论文图件、图注、脚本、metadata和审计报告。\n未复制全部原始大型实验数据，部分R2缓存按provenance路径复用。\nPython环境：/root/miniconda3/envs/g4splat/bin/python。\nE1重画：tools/build_cvpr_r3_e1_localization.py draw。\nE2/E3重现：运行对应build_cvpr_r3_e2_geometry.py / build_cvpr_r3_e3_compact.py。\nE4重画：tools/build_cvpr_r3_e4_prior.py draw。\nE5重画：tools/build_cvpr_r3_e5_deployment.py。\n汇总打包：tools/package_cvpr_r3_experiment_figures.py。\nprint_checks.json记录尺寸/字体/边界/PNG分辨率；manifest.json为文件SHA256。\n前两轮产物均保留。\n')
    files = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    manifest = {'date': '2026-10-03', 'revision': 3, 'status': 'five_figure_review_set',
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
