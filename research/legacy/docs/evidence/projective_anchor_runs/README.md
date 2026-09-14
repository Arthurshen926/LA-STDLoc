# Projective Anchor experiment record archive

This directory stores all small JSON/text records produced by the ShopFacade
and Stairs Gaussian-supported Projective Anchor round.  Per-query
`results.json` files are deterministically gzip-compressed.  Large tensors,
caches, RGB/depth renders and model weights remain outside Git and are bound by
path and SHA in the reports and manifest.

`manifest.json` contains the original source path, byte size and SHA-256 for
every record plus the archive path and digest.  Verify both archived bytes and
the current external sources with:

```bash
python -m scripts.archive_rendered_track_records \
  --output docs/evidence/projective_anchor_runs \
  --verify --check-sources
```

Invalid preflights and failed engineering launches are excluded from this
scientific archive and summarized in the result document where relevant.
