# Method

## Representation

An RGB Gaussian reconstruction supplies a frozen surface scaffold, camera-space
visibility, and raster contribution lineage. A Gaussian primitive is not
assumed to be a localization landmark. LaFGS instead constructs track-centric
anchors with three distinct identities:

- **Track identity:** a reciprocal, cycle-consistent set of real-image
  SuperPoint observations.
- **Anchor geometry:** a robust triangulation in the mapping coordinate frame.
- **Gaussian lineage:** the source primitives that support the track in raster
  space.

One rendering primitive can support multiple localization anchors, and one
anchor can retain lineage to multiple contributing primitives.

## Initialization

KCS selects primitives repeatedly supported by native keypoints across distinct
mapping views and trajectory bins. GWFF initializes each selected descriptor by
geometry-weighted fusion with robust cosine trimming. These steps initialize the
field; they are not the final learning objective.

The resulting wide scaffold uses 48K only as a safety cap. If the KCS gates
produce fewer eligible primitives, adaptive V2 stops at that consensus
saturation count and never fills the gap with non-consensus primitives. It
then runs a mapping-query-epoch-calibrated schedule of self-localization-guided
descriptor reconstruction. Current global candidates produce keep, swap, miss,
and false-attractor outcomes; bounded trust and protected local-peak terms keep
already precise correspondences stable. This Stage-A state is the A0 baseline
and the descriptor source used to construct Track-First evidence.

## Evidence and topology

Track-First matching builds cross-view tracks before assigning Gaussian
lineage. Robust triangulation rejects low-parallax and high-reprojection tracks.
The canonical candidate universe combines these independent track anchors with
Gaussian-supported coverage candidates.

Topology distillation retains quality-ranked tracks until the p10 per-query
anchor-keypoint bipartite matching rank reaches a mapping-derived target. A
mixed pool of leftover tracks and Gaussian anchors then closes feasible query
coverage. The final reserve is selected by dynamic, task-scaled full-SE(3)
D-optimal gain plus additive image/depth/spatial diversity. Query rows have
unit capacity with augmenting-path reassignment, the Hessian is updated after
every addition, source lineage and spatial voxel capacity are separate, and all
fixed counts are safety caps. Fisher weights combine candidate reliability with
query-specific detector repeatability. New track payloads propagate the full
triangulation covariance through the image Jacobian; old payloads use an
explicit isotropic trace fallback. Pose-reserve selection stops on normalized
objective gain and records logdet and translation-uncertainty curves.

The standard monotone-submodular guarantee applies to the cardinality-only
D-opt objective. Source and spatial capacities are practical anti-redundancy
safeguards; no `1-1/e` claim is made for their constrained implementation.

## Compact metric refresh

After topology distillation, a bounded low-rank metric is shared by mapping
queries and map descriptors. Complete-positive retrieval, current-map hard
outcomes, and trajectory-group DRO train this final A1 stage for a fixed number
of mapping-query epochs rather than a dataset-dependent fixed step count.
This short refresh preserves the single-descriptor compact map and uses the
same self-localization evidence as deployment. Geometry and RGB appearance
remain frozen in both descriptor-reconstruction stages.

## Deployment

The query image is processed once by native SuperPoint. Every descriptor takes
its global cosine top-1 anchor without landmark or keypoint caps. All resulting
2D-3D correspondences enter one standard PoseLib absolute-pose RANSAC solve.
The fixed pixel convention adds `0.5` to grid-index keypoints before PnP.

PnP uses one mapping-only scene calibration shared with self-localization
training. Its threshold is the larger of the focal-normalized angular gate and
the 97.5th percentile of per-track p90 reprojection residuals among stable
triangulated tracks. The residual term is capped at 12 pixels so a heavy-tailed
track set cannot silently relax the solver; test queries never enter this
calibration.
Detector density follows processed image area. Reprojection, epipolar, teacher,
and PnP thresholds follow focal/angular scale, while metric geometry thresholds
use the mapping track-graph baseline. The same resolved threshold contract is
asserted by function graph, raster provenance, positive teacher, compact
trainer, and deployment. A preliminary-to-track scale drift above the global
policy threshold triggers one Track-First evidence rebuild. The
5 cm/5 degree pose task scale remains fixed across datasets.

Resolved configuration separates focal/baseline and mapping-density dependent
scene parameters, globally shared method-policy constants, and
hardware/resource safety caps. Caps may terminate computation; they do not
define a required map size or add otherwise ineligible landmarks.
