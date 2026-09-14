"""Compose measured audit renders; no image synthesis or appearance adjustment."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import get_data_path


ROOT = Path('/mnt/pool/sqy/anygsloc_fast_prior_20260904')
OUT = ROOT / 'visual_comparison_20260907'
FONT = str(Path(get_data_path()) / 'fonts/ttf/DejaVuSans.ttf')


def panel(scene, audits, rows, destination, crop=False):
    reports = [json.loads((ROOT / scene / directory / 'report.json').read_text())
               for directory, _ in audits]
    names = [[r['image_name'] for r in report['records']] for report in reports]
    assert all(n == names[0] for n in names), 'Comparison requires identical views/order'
    assert all(r['dataset'] == reports[0]['dataset'] and r['split'] == 'test'
               for r in reports)
    width, height = 480, 270
    top, row_height = 115, height + 32
    canvas = Image.new('RGB', (width * (len(audits) + 1), top + row_height * len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 19)
    small = ImageFont.truetype(FONT, 15)
    draw.text((10, 7), f'{scene} | TEST views | ' + ('same central 40% crop' if crop else 'same cameras, no appearance adjustment'), fill='black', font=font)
    draw.text((10, 40), 'Reference RGB', fill='black', font=font)
    for col, ((_, title), report) in enumerate(zip(audits, reports), 1):
        x = width * col + 10
        draw.text((x, 40), title, fill='black', font=font)
        m = report['median']
        draw.text((x, 69), f"12-view median: {m['psnr_db']:.2f} dB / SSIM {m['ssim']:.3f}", fill='black', font=small)
        draw.text((x, 90), f"Background: {report['background']}", fill='black', font=small)
    for row, index in enumerate(rows):
        parts = []
        for col, (directory, _) in enumerate(audits):
            with Image.open(ROOT / scene / directory / f'view_{index:03d}.jpg') as im:
                w, h = im.size
                assert w % 3 == 0
                if col == 0:
                    parts.append(im.crop((0, 0, w // 3, h)))
                parts.append(im.crop((w // 3, 0, 2 * w // 3, h)))
        y = top + row * row_height
        draw.text((10, y), names[0][index], fill='black', font=small)
        for col, im in enumerate(parts):
            if crop:
                w, h = im.size
                im = im.crop((round(w*.3), round(h*.3), round(w*.7), round(h*.7)))
            canvas.paste(im.resize((width, height), Image.Resampling.LANCZOS), (col * width, y + 27))
    canvas.save(OUT / destination)
    return {'scene': scene, 'sources': [str(ROOT / scene / d / 'report.json') for d, _ in audits],
            'rows': rows, 'central_crop': crop, 'file': str(OUT / destination)}


def main():
    OUT.mkdir(exist_ok=True)
    audits = [('view_audit_long_lrm_raw_test12', 'Long-LRM raw (feed-forward)'),
              ('view_audit_long_lrm_ft1000_test12', 'Long-LRM + 1k optimization'),
              ('view_audit_test12_globaljoint_d1_ft1000', 'i3DGS + 1k (NOT feed-forward)')]
    records = []
    for filename, rows, crop in [('shopfacade_overview.png', [0, 4, 7, 11], False),
                                 ('shopfacade_crops.png', [0, 4, 7, 11], True),
                                 ('shopfacade_all12.png', list(range(12)), False)]:
        records.append(panel('ShopFacade', audits, rows, filename, crop))
    for scene in ['KingsCollege', 'OldHospital']:
        records.append(panel(scene, [('view_audit_anysplat_existing_test12_20260907',
                                      'AnySplat window-aligned (existing)'),
                                     ('view_audit_anysplat_allviews_test12_20260907',
                                      'AnySplat all mapping views')],
                             [0, 4, 7, 11], f'{scene.lower()}_anysplat.png'))
    (OUT / 'manifest.json').write_text(json.dumps({'note': 'Stored audit JPEGs; metrics from float render reports. AnySplat scenes are NOT a head-to-head model comparison.', 'panels': records}, indent=2) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
