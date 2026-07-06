# LaFGS Mainline Design

## Goal

Implement the new LaFGS mainline as localization-aware Feature Gaussian reconstruction, not as a post-hoc STDLoc overlay. RGB 3DGS remains a geometry, visibility, depth, and synthetic-view bootstrap; the trainable output is a localization-oriented Gaussian feature field.

## Scope

This pass introduces a tested reconstruction layer that can be used by training scripts without rewriting the whole renderer or evaluator:

- MVInit: initialize per-Gaussian localization descriptors from multi-view projected image features.
- LocRec: keep the existing direct landmark and full-bank retrieval losses as the feature-reconstruction core.
- DiffPnP-Loc: add 3D-to-soft-2D correspondences and a differentiable weighted PnP loss that can backpropagate to descriptors and, when explicitly enabled, small geometry residuals.
- Curriculum configuration: define the A-F stages from the attachment as code-level policy so training can start with frozen geometry and only later unlock pose loss or geometry residuals.

## Architecture

`localization_training/lafgs_reconstruction.py` owns method-level utilities and dataclasses. It depends on the existing direct landmark projection/sampling helpers and pose refiner, but does not depend on `stdloc.py`. Existing STDLoc code remains usable as an evaluator and baseline.

The implementation reuses the existing localization topology controller, but changes the signal path: DiffPnP now writes confidence, entropy, GT reprojection residual, and pose information into per-Gaussian localization stats. Split scoring consumes ambiguity, PnP residual, repeatability, projected footprint, confidence, and pose information.

## Data Flow

1. Load or train RGB 3DGS and initialize `GaussianModel` geometry from it.
2. Run MVInit over real training views: project each Gaussian, filter visibility/depth/alpha, sample image features, and aggregate robust normalized descriptors plus reliability.
3. Train LocRec losses with frozen RGB geometry: direct descriptor, multiview memory, full-bank retrieval, and optional feature render consistency.
4. Enable DiffPnP-Loc after correspondence quality is stable: Gaussian descriptors match into query feature maps with 3D-to-soft-2D soft-argmax, then weighted differentiable PnP computes pose and reprojection losses.
5. Only after pose loss is stable, allow geometry gradients or residual updates under existing geometry-anchor constraints.
6. In the topology phase, split candidates are selected from pose-aware stats; pruning remains opt-in and protected by the existing loc-opacity evidence gate.
7. Synthetic views are allowed for both dense and direct LocRec episodes. The LaFGS default uses RGB render -> frozen backbone features as the synthetic query target, and direct synthetic descriptor losses are downweighted through `synthetic_view_desc_weight`.

## Safety Constraints

- Geometry is frozen by default.
- DiffPnP uses local-window soft matching by default when GT pose is available to avoid early global averaging.
- PnP pose loss can be enabled independently from geometry gradients.
- Geometry residuals are bounded relative to RGB Gaussian scale through `bounded_geometry_residual_loss`.
- Synthetic RGB is not treated as a replacement for real feature supervision; `lafgs_synthetic_feature_source=rgb` is a bootstrap target path, while `loc_feature` remains a compatibility mode.
- Existing STDLoc evaluation remains unchanged.

## Testing

Unit tests cover MVInit aggregation, soft correspondence behavior, PnP loss gradients to descriptors, optional geometry gradients, and curriculum policy transitions.
