# Rendered Track method-enhancement convergence

## Outcome

V1.4 remains the source-image-free experimental baseline. This round implemented
the remaining three bounded method enhancements from the review—conditional
artifact-aware observation fusion, a genuine leave-one-query-observation-out
A1 metric refresh, and a full broad-Track completion oracle. None provides a
safe replacement across ShopFacade and Stairs. These experiments are evidence,
not a new gate, and they do not change the shared default.

The exact machine-readable lineage and results are in
[`docs/evidence/rendered_track_conditional_loo_completion_result.json`](evidence/rendered_track_conditional_loo_completion_result.json).

## Conditional artifact-aware fusion

The implementation no longer treats artifact stability as a scalar descriptor
weight. It removes an observation only when artifact reliability is low and its
descriptor is a Track outlier, while protecting identity-certified evidence,
unique pose-view support, unique mapping-sequence support, and a maximum 20%
trim per Track. Track membership, xyz, selected rows, map size, and query rows
remain fixed.

ShopFacade trimmed 7,910/168,948 observations (4.68%). Its frozen three-seed
test P90 improved from 8.057 cm to 7.431 cm and 5 cm recall from 82.201% to
83.172%, with no catastrophic cases, although median and raw precision were
slightly worse. Stairs trimmed 48,226/433,140 (11.13%): catastrophic count
improved by one per seed and 2 cm recall by 0.133 percentage points, but mean
TE rose from 10.926 cm to 11.078 cm, P90 from 9.335 cm to 14.980 cm, and 5 cm
recall fell from 82.967% to 82.233%. The rule is therefore not cross-scene safe.
No threshold scan was performed.

## LOO-aware A1

The metric training path now uses the full mapping trajectory without formal
cross-fit, excludes the current mapping query observation from every affected
Track bank, applies the learned metric consistently to query and temporary map
descriptors, and preserves pose-view bins separately from training/DRO groups.
The fixed run used 175 steps, rank 16, and residual norm at most 0.05.

ShopFacade was effectively neutral. On Stairs, mapping-only mean TE/CVaR95
worsened to 7.546/143.147 cm and the 21 catastrophic queries were unchanged.
This is sufficient to stop longer training; no test evaluation was run.

## Full broad-Track completion oracle

The nondeployable mapping-only oracle expands 5,788→14,753 anchors on
ShopFacade and 5,811→15,072 on Stairs. ShopFacade shows real capacity headroom
(median 0.251 cm, P90 0.581 cm), but its two mapping catastrophes remain.
Stairs gains raw match precision yet has worse median, P90, P95, recall, and the
same 21 catastrophes. Thus a simple lack of candidate anchors is not the cause,
and the proposed lazy/targeted completion should not be promoted.

## Diagnosis and next boundary

The remaining Stairs failures are best described as structured false Track and
pose consensus: coherent wrong correspondences survive aggregation and PoseLib.
They are not repaired by a scalar artifact score, this bounded residual metric,
or a larger candidate pool. Further work should change the correspondence or
geometric-consensus model only if a genuinely different hypothesis is proposed;
this enhancement family is converged.

