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

## Evidence and topology

Track-First matching builds cross-view tracks before assigning Gaussian
lineage. Robust triangulation rejects low-parallax and high-reprojection tracks.
The canonical candidate universe combines these independent track anchors with
Gaussian-supported coverage candidates.

Topology distillation first retains a quality-ranked Track core, then adds the
smallest configured coverage reserve needed for mapping-query support and pose
diversity. The output stores one descriptor and one 3D point per active anchor.

## Descriptor reconstruction

The current map repeatedly performs the same global sparse retrieval used at
deployment. Mapping poses label the current top candidates as:

- **keep:** current top-1 is a valid geometric positive;
- **swap:** another retrieved candidate is the correct positive;
- **miss:** no positive reached the candidate set;
- **reject:** a projected candidate is a false attractor.

The frozen Stage-A objective combines keep/swap/miss ranking, global-attractor
suppression, bounded descriptor trust, local peak preservation, and a
high-precision current-map protection set. Geometry and RGB appearance remain
frozen.

After topology distillation, a bounded low-rank metric is shared by mapping
queries and map descriptors. Complete-positive retrieval, current-map hard
outcomes, and trajectory-group DRO train this final A1 stage for 175 steps.

## Deployment

The query image is processed once by native SuperPoint. Every descriptor takes
its global cosine top-1 anchor without landmark or keypoint caps. All resulting
2D-3D correspondences enter one standard PoseLib absolute-pose RANSAC solve.
The fixed pixel convention adds `0.5` to grid-index keypoints before PnP.
