# V4 simplification record archive

This directory contains the 18 small JSON records from the cycle-core and
fixed-camera point-refinement mapping-only experiments. Large tensor maps,
Track payloads and caches remain outside Git and are referenced by path and
SHA-256 in the records and result evidence.

Verify the checked-in records and their live sources with:

```bash
python -m scripts.archive_rendered_track_records \
  --output docs/evidence/projective_anchor_v4_simplification_runs \
  --verify --check-sources
```

Neither experiment read test queries. Zero-byte stderr streams are reported
in the result evidence but are not included because the archive accepts only
small JSON/Markdown/text records.
