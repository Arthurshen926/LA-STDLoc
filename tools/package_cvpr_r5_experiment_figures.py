"""Package revision-five publication figures without changing experiment artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import subprocess
import re
import zipfile

sys.path.insert(0, '/root/.cache/anygsloc_figure_deps')
import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/experiments_cvpr_r5_20261003'
SPECS = [
    ('E1_final_localization', 'Coarse-to-fine localization', 'build_cvpr_r5_e1_localization.py'),
    ('E2_geometry_source_evidence', 'Geometry consistency', 'build_cvpr_r5_e2_geometry.py'),
    ('E3_compact_map', 'Compact landmark maps', 'build_cvpr_r5_e3_compact.py'),
    ('E4_gaussian_prior', 'Gaussian-prior comparison', 'build_cvpr_r5_e4_prior.py'),
    ('E5_deployment', 'Accuracy versus deployment cost', 'build_cvpr_r5_e5_deployment.py'),
]
CAPTIONS = json.loads((OUT / 'publication_captions.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build_context_proof():
    """Physical-size placement proof with real captions and resolved table references."""
    preamble = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.7in,textwidth=7.1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{graphicx,caption,booktabs,array,amsmath,float}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\begin{document}
"""
    body = [preamble]
    for i, ((stem, _, _), caption) in enumerate(zip(SPECS, CAPTIONS)):
        if i:
            body.append(r'\clearpage')
        body += [r'{\small\sffamily Physical-size figure proof · revision 5\hfill E' + str(i+1) + '}', r'\par\vspace{10pt}']
        if stem.startswith('E2'):
            body += [r'\begin{minipage}[t]{3.45in}',
                     rf'\includegraphics[width=\linewidth]{{{stem}.pdf}}',
                     r'\captionof{figure}{' + caption + '}', r'\end{minipage}']
        else:
            body += [rf'\includegraphics[width=\linewidth]{{{stem}.pdf}}',
                     r'\captionof{figure}{' + caption + '}']
    body += [r'\clearpage', r'\section{Supporting context}',
             'The following existing tables resolve the figure references in this placement proof. They are supporting context, not new experiments or additional figure panels.']
    for name in ['geometry', 'selection', 'priors']:
        table = (OUT / 'supporting_tables' / f'{name}.tex').read_text()
        # Keep support tables beneath their heading in this one-column proof.
        # The original insertion snippets retain their ordinary float settings.
        body.append(table.replace('{table*}', '{table}').replace('[!t]', '[H]'))
    body += [r'\clearpage', r'\input{publication_protocol}', r'\end{document}']
    source = OUT / 'LAYOUT_REVIEW.tex'
    source.write_text('\n'.join(body) + '\n')
    run = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', source.name],
                         cwd=OUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = (OUT / 'LAYOUT_REVIEW.log').read_text(errors='replace') if (OUT / 'LAYOUT_REVIEW.log').exists() else run.stdout
    errors = [line for line in log.splitlines() if ('undefined' in line.lower() and ('Reference' in line or 'references' in line)) or 'Overfull \\hbox' in line or 'Overfull \\vbox' in line]
    receipt = {'exit_code': run.returncode, 'reference_or_overflow_errors': errors,
               'geometry': '7.1-inch full-width figures; E2 at 3.45 inches. This is a placement proof, not a conference submission template.',
               'caption_word_counts': {s[0]: len(re.sub(r'\\[A-Za-z]+', '', c).split()) for s, c in zip(SPECS, CAPTIONS)}}
    (OUT / 'layout_review_checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if run.returncode or errors:
        print(run.stdout[-9000:])
        raise RuntimeError(receipt)


def main():
    for (stem, _, _), caption in zip(SPECS, CAPTIONS):
        path = OUT / f'{stem}.provenance.json'
        prov = json.loads(path.read_text())
        prov['revision'] = 5
        prov['publication_caption_latex'] = caption
        path.write_text(json.dumps(prov, indent=2) + '\n')
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
    review.set_metadata({'title': 'AnyGSLoc — experiment figures, revision 5',
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
    lines = ['REVISION 5 · Five publication figures', '',
             'E1: shared structural corner guides',
             'E2: frozen ECDF + localization control',
             'E3: architectural details, fewer labels',
             'E4: grouped priors, 8.3% less height',
             'E5: explicit native timing scope', '',
             'Vector PDF / SVG + high-resolution PNG',
             'Linked captions and visualization protocol',
             'Existing experiments; refined evidence']
    for j, line in enumerate(lines):
        draw.text((x + 30, y + 45 + j * 48), line, font=small, fill='#263341')
    canvas.save(OUT / 'CONTACT_SHEET.png')

    tex = ['% Requires graphicx; copy PDFs and this file into your paper and adjust paths.',
           '% Figure insertion snippets, not a standalone LaTeX document.',
           '% References require tab:geometry, tab:selection, tab:priors and sec:figure-protocol.',
           '% Existing table snippets are supplied in supporting_tables/.',
           '% Include publication_protocol.tex once in the paper or supplement.', '']
    for (stem, _, _), caption in zip(SPECS, CAPTIONS):
        env = 'figure' if stem.startswith('E2') else 'figure*'
        tex += [rf'\begin{{{env}}}[t]', r'  \centering',
                rf'  \includegraphics[width=\linewidth]{{{stem}.pdf}}',
                '  \\caption{' + caption + '}', rf'  \label{{fig:{stem.lower()}}}',
                rf'\end{{{env}}}', '']
    (OUT / 'figures.tex').write_text('\n'.join(tex))
    (OUT / 'CAPTIONS.txt').write_text('\n\n'.join(title + '\n' + caption
                                    for (_, title, _), caption in zip(SPECS, CAPTIONS)) + '\n')
    build_context_proof()
    sources = OUT / 'source'
    sources.mkdir(exist_ok=True)
    scripts = [s[2] for s in SPECS] + ['build_cvpr_e1_localization.py', 'build_cvpr_e4_prior.py',
              'figure_cache_reader.py', 'build_cvpr_r2_e1_localization.py', 'build_cvpr_r2_e2_geometry.py',
              'build_cvpr_r2_e3_compact.py', 'build_cvpr_r2_e4_prior.py', 'build_cvpr_r3_e1_localization.py', 'build_cvpr_r3_e2_geometry.py',
              'build_cvpr_r3_e3_compact.py', 'build_cvpr_r3_e4_prior.py', 'build_cvpr_r4_e1_localization.py', 'build_cvpr_r4_e2_geometry.py', 'build_cvpr_r4_e3_compact.py', 'build_cvpr_r4_e4_prior.py', 'build_cvpr_r4_e5_deployment.py', 'package_cvpr_r5_experiment_figures.py']
    for script in scripts:
        shutil.copy2(ROOT / 'tools' / script, sources / script)
    (OUT / 'README.txt').write_text((OUT / 'PACKAGE_NOTES.txt').read_text())
    files = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json' and p.suffix not in ['.aux', '.log', '.fdb_latexmk', '.fls', '.out']]
    manifest = {'date': '2026-10-03', 'revision': 5, 'status': 'five_figure_review_set',
                'no_new_algorithm_experiments': True, 'no_new_scene_rendering': True, 'figures': [s[0] for s in SPECS],
                'file_sha256': {str(p.relative_to(OUT)): sha(p) for p in files}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files + [OUT / 'manifest.json']:
            z.write(p, str(Path(OUT.name) / p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    publication = OUT.parent / (OUT.name + '_publication.zip')
    publication_files = [p for p in files if 'assets' not in p.relative_to(OUT).parts and 'before' not in p.relative_to(OUT).parts and p.suffix not in ['.aux', '.log', '.fdb_latexmk', '.fls', '.out']]
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
