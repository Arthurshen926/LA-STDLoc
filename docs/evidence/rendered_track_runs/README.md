# Render-only run-record archive

This directory preserves the small, human-auditable records behind the
Stairs/ShopFacade render-only method history.  It covers the initial feasibility
probe through V1.5 deployment-feedback closure, including the full-reference
Stairs replay.

The archive contains 679 source records from eleven experiment roots.  Their
original size is 99,484,413 bytes; the checked-in archive is 22,474,988 bytes
before Git pack compression.  Per-query `results.json` files are stored as
deterministic `results.json.gz`; other JSON and exit-status text files retain
their original bytes.  A few historical `*_invocation.json` files are actually
terminal transcripts rather than JSON documents; their exact bytes are kept
under a `.json.log` name and identified as `text/plain` in the manifest.  Large
`.pt` artifacts, feature caches, renders, images, and model weights are
deliberately excluded and remain identified by path and SHA in the contracts
and result evidence.

## Round index

| Archive directory | Method stage | Human-readable conclusion |
|---|---|---|
| `v10_render_probe` | rendered-RGB geometry feasibility | [initial experiment](../../rendered_rgb_track_only_experiment.md) |
| `v10_mapping_audits` | mapping-only retrieval/geometry audits | [initial experiment](../../rendered_rgb_track_only_experiment.md) |
| `v10_fullchain` | first complete Track-only localization chain | [initial experiment](../../rendered_rgb_track_only_experiment.md) |
| `v11_appearance` | support-aware appearance and A1 controls | [V1.1 section](../../rendered_rgb_track_only_experiment.md) |
| `v12_support` | bounded support-certified Track repair | [V1.2 section](../../rendered_rgb_track_only_experiment.md) |
| `v13_component_repair` | component-rebuilt corrective audit | [V1.3 section](../../rendered_rgb_track_only_experiment.md) |
| `v14_fullmap` | full-mapping selector and V1.4 baseline | [V1.4 evidence](../rendered_rgb_track_fullmap_v14.json) |
| `r1_artifact_stability` | raw/clean artifact-stability ablation | [R1 result](../../rendered_rgb_track_artifact_stability_result.md) |
| `v14_method_enhancements` | conditional fusion, LOO-A1, completion | [enhancement result](../../rendered_track_conditional_loo_completion_result.md) |
| `full_reference_history_replay` | historical arms on rebuilt Stairs prior | [full-reference replay](../../rendered_track_full_reference_history_revalidation.md) |
| `full_reference_v14_v15` | rebuilt-prior V1.4 and closed-loop V1.5 | [V1.5 result](../../rendered_track_pose_feedback_closed_loop_result.md) |

The concise machine conclusions remain in `docs/evidence/*.json`; this archive
adds the underlying reports, contracts, summaries, invocations, exit statuses,
and per-query JSON results.  Invalid preflight/smoke launches are intentionally
excluded because they are not scientific evidence.  Their existence and
effect, where relevant, remain documented in the concise result files.

## Integrity

[`manifest.json`](manifest.json) records for every file:

- the original absolute path, byte size, and SHA-256;
- the repository-relative archive path, byte size, and SHA-256;
- whether the archived bytes are identical or deterministic gzip.

The archive can be verified without the external experiment roots:

```bash
python scripts/archive_rendered_track_records.py --verify
```

When the original `/mnt/pool` records are still present, also verify that they
have not drifted:

```bash
python scripts/archive_rendered_track_records.py --verify --check-sources
```

To inspect a compressed per-query result without modifying it:

```bash
gzip -cd docs/evidence/rendered_track_runs/<stage>/<scene>/<run>/results.json.gz \
  | python -m json.tool
```
