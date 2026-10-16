# Method and implementation

The latest paper operating point is represented by explicit dataset profiles:

| Profile | Gaussian prior | Observation extraction | Query masking | Main benchmark row |
|---|---|---|---|---|
| Cambridge | vanilla 3DGS | unmasked SuperPoint top-2048, then V2 observation filtering | semantic/geometric valid mask | Full |
| 7Scenes / 12Scenes | 2DGS | alpha-gated rendered observation extraction, then filtering | none | Compact v2 |

These choices preserve the experiments actually reported in the current manuscript. The indoor Compact results must not be relabeled as Full. Scene calibration determines the online keypoint count and reprojection threshold from mapping cameras; it never reads test poses.

| Paper component | Implementation |
|---|---|
| Rendered observations and certificates | `pipeline/build.py`, `mapping/render_certificate.py` |
| Multi-view matching and feature tracks | `mapping/projective_association.py`, camera-pair policies |
| Robust ray triangulation | `mapping/projective_reconstruction.py`, `mapping/triangulation.py` |
| Depth-guided track augmentation | `mapping/projective_completion.py` |
| Track descriptor fusion | `features/multiview_fusion.py` |
| Quality, coverage, pose-aware selection | `mapping/compact.py`, `mapping/selector.py`, `mapping/pose_information.py` |
| Support-preserving terminal cleanup | `mapping/compact_v2.py`, `mapping/post_pose_cleanup.py` |
| Sparse matching and one pose solve | `online/runtime.py`, `online/matching.py`, `online/pose.py` |
| Paper experiment composition | `pipeline/`, `reproduction/run.py` |

Full and Compact share the retained coordinates and descriptors. Selection uses mapping-view support only. The final cleanup removes redundant coverage additions while preserving capped per-reference feature-row support; it is the terminal part of the selector, not an additional pose solver.

The online defaults are 256-dimensional SuperPoint descriptors, similarity-based matching, a 0.65 retention fraction with at least 256 retained matches when available, and one PoseLib solve. Confidence is 0.999, the minimum RANSAC budget is 250, and the maximum is 30,000. The exact resolved configuration is saved with every evaluation. The historical schema name `compact-fast-v1` in online records identifies this solver operating point; it does **not** select the old map cleanup. Current mapping calls `select_compact_rows_v2` explicitly.

Numerical kernels were retained from the experiment code. Dataset/ablation adapters have been consolidated into a small pipeline instead of importing dated experimental directories. `reproduction/source_lineage.json` records the origin hashes. Changes to numerical behavior should be checked against the reference records and real-map parity, not accepted on the basis of import tests alone.
