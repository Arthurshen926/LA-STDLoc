# ShopFacade virtual Track augmentation: final No-Go

The frozen 7,854-Anchor ShopFacade map was evaluated exactly once after
mapping-only construction was complete.  The test protocol was global Top-1,
one standard PoseLib call per query, seed 2026, and no retrieval, refinement,
group-aware pose, guided sampling, or assignment.  Test queries were used only
for this final frozen evaluation; they never participated in candidate
generation, map selection, calibration, the mapping-camera oracle, or the
support-ring probe.

The augmented map obtained 2.103 cm / 0.118 deg median error, 47.57% R2, and
82.52% R5 on 103 ShopFacade test queries.  Ours-G remains better at 2.039 cm /
0.113 deg and 49.51% R2 with the same R5.  Per query, augmentation improved 47
translation errors and regressed 56; it gained no R2 query and lost two.  The
G+RGB-Descriptor oracle remains stronger in median, P90, R2, R5, and angular
metrics.  The augmented mean translation error is 0.040 cm lower than that
oracle, but this isolated mean does not overturn the broader regression.
Historical runtime numbers were produced under different exact-acceleration
and runtime conditions, so no map-attributable speed claim is made.

The mapping-only failure evidence explains the No-Go.  The formal and
augmented maps both have zero GT-visible Anchors for `seq2/frame00036.png` and
`seq2/frame00043.png`.  A symmetric leave-one-camera-family-out oracle over all
231 mapping cameras and all 256 real-supported candidates can repair only the
second tail.  A preregistered support-ring upper bound then generated the same
six local directions and 0.20/0.35/0.50 m radii for every formal L1-zero
camera.  It still produced zero multi-family visible or detector-accessible
evidence for frame 36.  Its candidate render sees a z-buffer surface near
0.184 m where the held-out render unprojects the feature rays near 1.64 m, a
roughly 1.47 m median cross-view depth disagreement.  This is a frozen
Gaussian-prior geometry limitation inside the tested safe envelope, not a
remaining selector problem.

Therefore the 7,854-Anchor augmented map must not replace Ours-G, the current
256-candidate selector should not be tuned further, and the support-ring probe
must not proceed to map construction or test.  The implementation stays
opt-in and the default pipeline is unchanged.  Compact hashes and source paths
for every full external report are recorded in
`docs/evidence/v4_sufficiency_virtual_track_final_shop_nogo.json`; no binary or
large per-query artifact is committed.
