# Validation of this source candidate

Validation was performed on October 15, 2026. The experiment records correspond to the paper's October 13 results; supplied figures follow the October 14 manuscript.

Completed checks:

- All 244 accepted scene/configuration/output cells, 35 PSNR cells, and 90 formal timing processes are indexed. Every localization group has its complete scene and query population.
- Recomputed statistics match **761 locally measured numeric entries in all 22 current paper tables**, at the precision printed in the manuscript. Literature rows remain cited reference values.
- Fresh Shop Facade extraction from the accepted vanilla 3DGS prior covers all **231 mapping views**. Keypoints, descriptors, scores, V2 masks, sampled depth/alpha, intrinsics, and poses match the accepted observation cache exactly.
- Rebuilt Shop Facade Full (**25,953 points**) and Compact (**12,014 points**) have exactly the same coordinates and descriptors as the accepted maps. All **103 query errors match exactly for both maps**, using the new entry points.
- Fresh PSNR on all 231 Shop Facade mapping images differs by at most **0.0000161 dB per image** from the accepted run, below the table's precision.
- Fresh Shop Facade center/depth/ray controls reproduce every landmark ID, coordinate, descriptor, and mapping-observation residual exactly.
- The new native STDLoc and ULF-Loc accuracy entries both reproduce all 103 Shop Facade initial/refined translation and rotation errors exactly, with matching ground-truth pose order.
- One fresh Shop Facade timing process each for AnyGSLoc Compact, STDLoc, and ULF-Loc completes the fixed 50 warm-ups and 100 measured queries. Fresh timing reporting validates the query subset and cumulative native stages. These three smoke rounds validate the new entry points; they do not replace the accepted 90-process latency results.
- Synthetic calibrated geometry, source/test separation, real-only auxiliary completion, support-preserving cleanup, metric failure denominators, deterministic timing subsets, query masks, and native adapter definition preservation are covered by the test suite.
- All 15 tests pass in the local environment. The code passes Ruff's standard checks and formatting. The wheel imports and runs the planning entry from an isolated installation outside the source directory; the source archive additionally includes configs, reproduction tools, records, documents, and editable figures.
- The five experimental figure layouts have CPU regeneration entries. Their accepted SVG sources and data bindings are retained.
- All 504 entries in the local asset layout are present, with no recorded-size mismatches. This presence check is distinct from rereading every large asset for a full hash verification.
- All five environment profiles pass their specified version/import checks and a small CUDA operation. The launcher is verified with main-pipeline planning and the native baseline command plan. Full Conda/pip snapshots are stored privately outside the source archive. This validates the existing installations, not a clean reinstall of all five environments.

Machine-readable evidence is in `reproduction/validation/`. The full local check outputs are kept outside the source distribution under `checks/`.

This task did **not** rerun every GPU experiment, retrain every upstream prior/native baseline, or repeat the 90-process formal timing campaign. The supplied reference results are independently reaggregated; fresh end-to-end equivalence is established on Shop Facade. Other scenes/producers have executable profiles and bound artifacts, but a fresh all-scene campaign remains a broader validation task. Repeated latency is expected to vary by host and load.

The public download bundles for datasets, priors, deployment maps, and native source/model assets have not been uploaded. The source and optional figure-input archives are prepared locally. Their manifests and the local asset exporter provide the concrete material needed for the publication step.
