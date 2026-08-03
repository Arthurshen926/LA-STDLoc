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
