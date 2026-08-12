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
