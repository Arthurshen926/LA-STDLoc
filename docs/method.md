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

The resulting wide 48K scaffold first runs 1K steps of self-localization-guided
descriptor reconstruction. Current global candidates produce keep, swap, miss,
and false-attractor outcomes; bounded trust and protected local-peak terms keep
already precise correspondences stable. This Stage-A state is the A0 baseline
and the descriptor source used to construct Track-First evidence.

## Evidence and topology

Track-First matching builds cross-view tracks before assigning Gaussian
lineage. Robust triangulation rejects low-parallax and high-reprojection tracks.
The canonical candidate universe combines these independent track anchors with
Gaussian-supported coverage candidates.

Topology distillation first retains a quality-ranked Track core, then adds the
smallest configured coverage reserve needed for mapping-query support and pose
diversity. The output stores one descriptor and one 3D point per active anchor.

## Compact metric refresh

After topology distillation, a bounded low-rank metric is shared by mapping
queries and map descriptors. Complete-positive retrieval, current-map hard
outcomes, and trajectory-group DRO train this final A1 stage for 175 steps.
This short refresh preserves the single-descriptor compact map and uses the
same self-localization evidence as deployment. Geometry and RGB appearance
remain frozen in both descriptor-reconstruction stages.

## Deployment

The query image is processed once by native SuperPoint. Every descriptor takes
its global cosine top-1 anchor without landmark or keypoint caps. All resulting
2D-3D correspondences enter one standard PoseLib absolute-pose RANSAC solve.
The fixed pixel convention adds `0.5` to grid-index keypoints before PnP.
