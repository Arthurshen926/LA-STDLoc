# Artifact Contract

Every cross-stage artifact has an explicit schema/version, ordered query or
anchor registry, parent hashes, resolved config hash, and producer worktree
state. `common/artifact_contract.py` verifies content rather than relying on
filenames.

Core artifacts are:

| Artifact | Required identity |
| --- | --- |
| normalized prior | primitive fields, type, SH degree, PLY SHA |
| query cache | ordered image names, processed resolution, frontend contract |
| Track payload | observation CSR, query bins, triangulated geometry |
| canonical map | anchor IDs, source primitives, track IDs, types |
| raster provenance | query rows, primitive IDs, contribution mass |
| function graph | candidate rows, legal flags, evidence version |
| compact map | stable anchor registry and topology lineage |
| metric checkpoint | exact compact-map row order and metric config |

## Neutral Anchor Registry materialization

`scripts/materialize_anchor_registry.py` writes an Evidence-Grounded Anchor
Registry as a **sibling audit artifact**.  It never replaces the trained
`lafgs_materialized_anchor_map` consumed by `SparseLocalizer`.

Every supplied parent is a paired `PATH + expected SHA-256`.  Standalone legacy
audits require the trained map and may explicitly omit unavailable evidence;
an unresolved legacy selection is accepted only with
`--allow-legacy-unresolved-audit` and is never pipeline eligible.  A new
pipeline uses `--require-pipeline-parents`, which requires the exact trained
map, compact map, positive teacher, Track payload, query cache, raster and
selection provenance, mapping-only scene calibration, metric state, resolved
config, and Gaussian PLY.  Their row/query/path relationships are validated in
addition to their file hashes.

The compatibility `registry_sha256` retains its V1 identity.  New contracts
also carry a framed `full_registry_sha256` and per-field hashes.  These bind
field name, value kind, tensor dtype, tensor shape, and bytes across the full
Registry schema, including observation CSR, identity, geometry, selection,
evidence, and all localization tensors.  The artifact is saved to a temporary
file, reloaded, checked for bitwise localization parity, and installed without
overwrite.  Its JSON contract is installed atomically last.  An artifact
without that completion contract is a failed partial run and must be
quarantined rather than resumed.

Large `.pt`, `.pth`, and `.ckpt` files are runtime outputs and are not committed.
`paper_baseline/` stores only compact fixtures, hashes, manifests, and reports.

## Zero-budget canonical maps

`topology/candidates.py --budget 0` materializes only the frozen base-anchor
prefix. The `--base_state`, `--track_payload`, and `--query_cache` paths remain
required declared context and must each name an existing file. Unless
`--evaluate_zero_budget_eligibility` is passed, the CLI deserializes only the
base state; Track and query files may contain unreadable bytes because their
contents cannot affect a zero-Track map. Eligibility is consequently reported
as unevaluated (`eligible_track_count: null`), not as zero.

Canonical-map provenance separates `declared_context` from
`materialized_dependencies`. Each dependency has a `used` flag; `used: false`
means that its values are not an input to the materialized tensor fields. It is
not a forensic assertion that no producer process ever opened or inspected the
file. The flat `base_state`, `track_payload`, and `query_cache` keys are retained
as declared-context compatibility aliases for downstream lineage checks.
