# Weak-prior controlled diagnostics: evidence, not a promised upper bound

**2026-09-08 后续勘误：** 本文以下为历史受控实验。随后发现
OldHospital/KingsCollege 的 Vanilla Gaussian 训练输入存在逐像素确认的
重复去畸变错误，固定地图补偿已带来明显收益，见
[接入错误与补偿验证](vanilla_rectification_bug_20260908.md)。
不能将本文差距直接视为 vanilla 重建算法或 Gaussian-only 方法的上限。

All new experiments below use the same-code OldHospital F0 maps, 182 test
queries, unchanged Base matching/PoseLib, seed 2026 unless noted. No offline
self-localization feedback is enabled. All normal candidate construction uses
only mapping Gaussian renders and given cameras. Real-image diagnostics are
explicitly isolated and must not enter the source-image-free method table.
These are test-adaptive development results, not independent generalization.

## 1. Fixed-geometry descriptor counterfactual — completed for both priors

`probe_v8_real_descriptor_ceiling` first replays original descriptor fusion
**bit exactly for every Anchor**. It recovers the original association-to-cache
row remap, base confidence, completion confidence, detector weights and view
bins. It then samples real **mapping** image descriptors at the frozen render
keypoints and repeats the same fusion. No redetection, Track selection, XYZ,
map capacity, association or fusion-weight change is allowed.

Independent tensor comparison confirmed that only `anchor_features`,
`anchor_observation_features` and `provenance` changed. This is not a perfect
appearance oracle: render keypoint offsets and existing wrong identities remain.

| Prior / descriptor | Median TE cm | Mean TE cm | P90 TE cm | Median RE deg | >=100 cm |
|---|---:|---:|---:|---:|---:|
| Vanilla / render F0 | 21.1187 | 31.1317 | 58.4321 | .281331 | 4 |
| Vanilla / real mapping diagnostic | 16.1070 | 24.7842 | 49.4686 | .241891 | 6 |
| MAtCHA / render F0 | 9.7066 | 19.2791 | 49.6244 | .171347 | 4 |
| MAtCHA / real mapping diagnostic | 9.6405 | 18.5540 | 42.2837 | .162898 | 3 |

Vanilla TE improves on 151/182 queries, worsens on 31. Paired-query bootstrap
(2,000 draws, RNG 20260907) gives a descriptive 95% interval of [-7.01,-3.18]
cm for the difference of medians. MAtCHA's interval is [-2.15,+1.12] cm.
This does not account for temporal dependence or adaptive hypothesis selection;
it is not a formal independent hypothesis test. The gap between the two
observed medians shrinks from 11.41 to 6.47 cm, but that ratio is **not** an
identified causal fraction attributable to appearance rather than geometry.

Supported: the current weak map has recoverable descriptor/matching headroom,
and the claim that its entire error is irrecoverable XYZ quality is too strong.
Not supported: render-only processing can necessarily reproduce this recovery,
or real descriptor replacement solves robustness (catastrophic failures rise).

Diagnostic maps carry `uses_source_mapping_rgb=true`, `deployment_eligible=false`.
The localizer rejects them without explicit `--allow-source-rgb-diagnostic`.
Evaluation summaries record the opt-in; contracts generated after the guard
update explicitly reject a source-image-free deployment claim for these runs.

## 2. Geometry-aware descriptor evidence alignment — two variants completed

F0's final geometric IRLS residual weights were not carried through descriptor
fusion. Moreover, legacy fusion normalizes each camera-bin prototype and then
averages retained bins equally. An absolute reliability factor can cancel in
that normalization, especially for singleton bins. This is a property of the
fusion rule, not proof of the cause of every earlier reliability experiment.

At the frozen final XYZ, compute Huber(2 px) reliability for **all** original
observations. This is not claimed to replay the exact <=32-row production IRLS
subset. Neither candidate changes XYZ, capacity, IDs or descriptor support.

- `within_bin`: multiply original observation weights by this reliability;
  leave legacy bin normalization unchanged.
- `across_bin`: additionally retain the original-weighted mean reliability of
  each bin when combining its prototype. Preserve bin balancing and trimming;
  do not hard-delete Anchors or let large bins dominate by count.

Vanilla has 7.59% of observation residuals above 4 px, accounting for only
4.52% of original pre-bin weight mass; MAtCHA has 6.41% / 3.86%. Most fused
descriptors change very little (Vanilla cosine P10 .99980 / .99965).

| Prior / fusion | Median TE cm | Mean TE cm | P90 TE cm | Median RE deg |
|---|---:|---:|---:|---:|
| Vanilla / within bin | 21.4070 | 31.2122 | 59.8869 | .288901 |
| Vanilla / across bin | 21.3325 | 30.4595 | 58.9154 | .290731 |
| MAtCHA / within bin | 9.5205 | 19.4285 | 48.9696 | .175017 |
| MAtCHA / across bin | 9.1009 | 19.3729 | 47.8979 | .174319 |

Both fail to close the Vanilla gap and are **not promoted**. MAtCHA's small
median TE gain does not establish a general gain: RE/mean TE are worse and
the descriptive paired-bootstrap median intervals cross zero. No threshold
sweep is warranted by these results. Online matcher cost remains about 5 ms;
no extra online module is introduced, nor is a real-time improvement proven.

## 3. Independent real-track geometry reference — implemented, insufficient

COLMAP `images.bin` has no retained 2D feature observations. The original NVM
does retain real-image tracks. `audit_nvm_geometry_reference` resolves JPEG to
PNG stem aliases only after validating mapping poses and intrinsics, then
undistorts measured image coordinates with the processed-camera model.
Reference reprojection P50/P90/P99 = 1.55/3.77/7.63 px for mapping observations.

An Anchor is associated to a real Track by at least three observed mapping
keypoints agreeing on its NVM identity, >=60% Track support, >=2 vote margin,
and >=.5 px nearest/second-nearest separation. Reference geometry must pass
spatial crossfit and is refit using only real **mapping** observations. Original
all-view SfM initializes that refit; this reference is diagnostic, not GT-free
production geometry or absolute truth. There are no query-GT nearest-point
assignments or manually verified identity claims.

| Association radius | Vanilla accepted | MAtCHA accepted | Shared reference IDs |
|---|---:|---:|---:|
| 2 px | 12 | 17 | 1 |
| 4 px | 163 | 141 | 19 |

At 4 px this covers only .213%/.193% of the maps. Their separate median XYZ
errors (8.93/8.46 cm) are highly selected and not a fair whole-map paired
comparison. **Do not use them to infer a geometric accuracy ceiling.**
This route has insufficient independent coverage for the requested fixed-
physical-correspondence pose comparison. It is neither confirmation nor
falsification of the broader identity-correction direction.

## 4. Fixed-descriptor geometry-source control — completed for both priors

`probe_v8_depth_geometry_control` replaces each XYZ by the coordinate median
of the same Track's backprojected cached axial render depths. Descriptors,
identities, selected observations and capacity remain fixed. All Anchors have
>=3 valid depths, so no ray fallback was used. Ray-derived covariance is removed
rather than mislabeled as valid for depth geometry; Base has no covariance gate.

This is a conditional depth-vs-ray baseline, **not** a full direct Gaussian-
primitive baseline, and not a controlled RGB-preserving prior corruption test.

| Prior / XYZ source | Median TE cm | Mean TE cm | P90 TE cm | Median RE deg |
|---|---:|---:|---:|---:|
| Vanilla / depth median | 19.9954 | 32.8940 | 58.8265 | .253941 |
| MAtCHA / depth median | 10.3429 | 21.0543 | 51.7022 | .175623 |

Vanilla's median improves but mean worsens; paired-query bootstrap median
difference interval [-3.42,+2.95] cm crosses zero. MAtCHA degrades. This does
**not** substantiate universal ray superiority or a deployable switch back to
depth. Preserve the contrary evidence rather than selecting only favorable
numbers for a prior-robustness story.

On these medians, weak/strong TE sensitivity is 2.176 for F0 ray geometry,
1.933 for the conditional depth baseline (absolute gaps 11.41/9.65 cm).
Thus this specific control does **not** support a claim that F0 has lower
prior-quality sensitivity. It does not isolate RGB changes or establish a
universal comparison between all ray and primitive/depth representations.

## 5. Multiview featuremetric observation correction — pilot completed

Render-only fixed SuperPoint fields; optimize each keypoint toward a descriptor
consensus from OTHER mapping views, with a 1.5 px displacement bound and trust
term. Then fixed-camera Huber triangulation and single-descriptor re-fusion.
Accept only if both spatial crossfit groups are nonworse, the maximum heldout
median improves >=5%, and original-measurement fit / XYZ update guards pass.
This is a one-pass stride-8-field pilot, not a replication of Pixel-Perfect SfM.

The 210 sampled unstable Tracks require 710 existing mapping views. Re-render
descriptor replay maximum absolute difference is 7.45e-8. 76 updates pass,
median/P90 XYZ shift 3.44/14.58 cm. Original source CSR remains traceable, with
corrected geometry/descriptor coordinates recorded as explicit overrides.

Final median TE remains 21.1187 cm; mean TE 31.13175 -> 31.13065 cm, median RE
.281331 -> .281296 degrees. TE improves on 31 queries, worsens on 13, unchanged
on 138. This is **not a meaningful final-pose improvement** and is not promoted.

Only 419 original matches and 58 original inliers involve the changed rows;
31 match identities change. On 57 fixed-ID original inliers, GT reprojection
median improves 6.84 -> 6.03 px. This limited positive local evidence motivates
one full-map target-selection test, not a threshold sweep or a claimed gain.

## 6. Second seed of the geometry-source control — completed

At seed 2027, Vanilla ray/depth medians again equal 21.1187/19.9954 cm and
median RE .281331/.253941 deg. Mean TE is 31.8464/29.5846 cm (unlike seed 2026,
where depth worsened the mean), and failures are 5/4. The median difference
survives these two solver seeds, but the tail is seed-sensitive and the paired
query interval still crosses zero. This is not multi-scene confirmation.

## 7. Gaussian surface geometry detection — full-view ablation completed

`surface_row_evidence` combines expected/median depth disagreement with the
angle between rendered primitive normals and depth-derived normals. Both are
world-frame in the installed gsplat implementation; sign is ignored. Require
finite positive depths, alpha >=.5 and normal norms >.1 before providing a vote.
Flag a row only if relative depth discrepancy >2% **and** normal-axis angle >45°.
Reject an Anchor only with >=3 such observations, >=half its original Track,
and >=2 conflicting camera bins. Unknowns are preserved.

All 895 mapping views per prior were rerendered; sampled expected depths were
bitwise equal to the frozen cache (maximum absolute delta zero). No real RGB
or test input is consumed. Both channels come from the same Gaussian scene;
this detector cannot identify a stable common world bias.

Vanilla flags 408/76,701 Anchors; MAtCHA 148/73,241. These rows have lower raw
GT-4px precision (2.41%/2.74%) than the remaining no-conflict rows (10.00%/14.72%),
so the internal signal is associated with difficult matches. But flagged rows
still contribute 47/40 original correct-under-4px PnP inliers.

| Filtered map | Median TE cm | Mean TE cm | P90 TE cm | Median RE deg |
|---|---:|---:|---:|---:|
| Vanilla | 21.1357 | 31.1998 | 62.0060 | .281331 |
| MAtCHA | 9.5482 | 19.4172 | 49.6244 | .171347 |

The specific hard gate **does not improve Vanilla localization** and is not
promoted. Keep the diagnostic, not the filter. Lower group precision does not
mean every member is useless, and removing a bad first candidate does not
guarantee its replacement is correct. This is an implemented, fully evaluated
geometric validity hypothesis, not a claim that all such detectors are useless.

The replacement audit makes the failure mechanism concrete. On Vanilla's 1,947
removed original matches, 47 were originally within GT 4 px; only 44 replacements
are within 4 px. There are 39 wrong-to-close and 42 close-to-wrong replacements.
No nonremoved match changes identity. Thus the gate mainly replaces one wrong
candidate with another; it does not recover the needed correct correspondences.
MAtCHA's analogous counts are 1,462 removed matches, 40 -> 63 within 4 px,
59 wrong-to-close and 36 close-to-wrong. These labels indicate GT projection
proximity, not independently proven physical identities.

## Implementation safeguards

- Source-RGB diagnostic opt-in is required at map load; default F0 unchanged.
- Original fusion is replayed exactly before descriptor interventions.
- Camera registry and per-view render parity are checked; deriving FOV from a
  rounded cached K was inadequate for exact render replay in the new pilot, so
  attested original camera FOV is used. This was a new-probe setup correction,
  not a discovered explanation of historical F0 errors.
- A nonfinite spatial-prediction comparison edge case was caught before the
  full-map correction produced artifacts. That run was interrupted and restarted
  with explicit finite-value checks. All 76 earlier pilot acceptances also pass
  the corrected check; the completed pilot's conclusions are unaffected.
- Post-adjustment descriptor targets span the camera groups. Consequently that
  crossfit check is internal stability evidence, not statistically independent
  validation; only the external final evaluation tests localization utility.

## 8. Full-map multiview correction — completed, not promoted

The full Vanilla audit labels 20,194 Anchors internally consistent, 15,575
unstable and 40,932 unknown. Apply the same correction to **all** 8,331 eligible
unstable base Tracks with 7–19 observations (92,711 observations, 892 views).
3,285 updates pass; median/P90 XYZ shift 5.52/27.74 cm. Nonselected XYZ and
descriptors remain bitwise unchanged. Original observation CSR is unchanged,
with adjusted coordinates in an explicit override ledger.

| Vanilla F0 -> full correction | F0 | Corrected |
|---|---:|---:|
| Median TE cm | 21.1187 | 21.2749 |
| Mean TE cm | 31.1317 | 30.4299 |
| P90 TE cm | 58.4321 | 61.4138 |
| Median RE deg | .281331 | .284685 |
| >=100 cm | 4 | 4 |

TE improves/worsens/unchanged on 92/88/2 queries. Mean improvement does not
outweigh worse median TE/RE and P90. Do not deploy this variant or sweep its
parameters. This is a negative full-target test, not proof against all
featuremetric refinement or the unresolved <7-observation Tracks.

The changed rows now affect 13,851 original matches and 2,112 original inliers;
1,371 match identities change. For 2,065 fixed-ID original inliers, GT projection
median is 4.442 -> 4.373 px; <=4 px count 923 -> 922. Meanwhile the median of
accepted Tracks' maximum spatial prediction errors is 3.180 -> 1.913 px.
This contrast directly shows that improving internal render consistency did
not yield commensurate real-query information. It does not independently
identify whether each residual error is wrong identity or systematic bias.

## Completion status and unresolved evidence

All bounded experiment runs described here are complete: 18 evaluations,
including two reused F0 reference evaluations and 16 newly executed full-test
evaluations. Each uses all 182 OldHospital queries. No unfinished run remains
behind these conclusions. The original 16-evaluation ledger is preserved as
`controlled_diagnostics_summary.json`; the final 18-evaluation ledger is
`controlled_diagnostics_summary_with_no_v2.json`.

Strong physical-identity verification remains unresolved: available NVM
corroboration has too few common points, and an automatic multi-view proximity
test cannot certify identity. Broader keypoint/identity repair is therefore
**not falsified**. A well-covered independently verified real-track reference
would be needed to close the proposed fixed-physical-correspondence diagnosis.
The present internal certificates cannot manufacture that missing evidence.
Arbitrary-quality prior invariance is not supported and should not be a paper
claim; the conditional auxiliary-depth implementation property is separate.

## Conditional auxiliary-geometry invariance

A regression test runs actual base `reconstruct_projective_anchors` on fixed
five-camera measurements of a known point, with identical Track/descriptor
evidence and view-dependent depth scales spanning .01x to 100x. XYZ, fused
descriptors and covariance are bitwise identical. This supports the conditional
implementation property: base ray reconstruction does not consume auxiliary
depth to determine its final XYZ when purification is off.

V2/completion and RGB/visibility are deliberately frozen in this test. The
result is not evidence that physically corrupting a Gaussian prior leaves the
complete method invariant; those upstream pathways can change observations.

## No-V2 control protocol notes

The optional `--disable-v2-row-filter` path retains all cached detector rows,
rebuilds associations and geometry/completion, and explicitly marks V2 disabled
in graph lineage, construction and report contracts. Default F0 behavior is
unchanged; reusing an association graph is prohibited for this ablation.
Both priors retain the same 3,101 camera-pair budget and 1,832,960 cached rows.
Native cached detector alpha/border handling is still present; this is no V2,
not the removal of every original image-validity check.

Anchor counts become 97,569 (Vanilla) and 87,677 (MAtCHA), versus F0's 76,701 and
73,241. This is an end-to-end construction ablation, not a fixed-capacity or
fixed-correspondence control. Geometry/feature effects and added coverage are
coupled. Gaussian depth remains used for completion proposals, not base XYZ.

Initial no-V2 reports retain historical generic strings such as
`row_budget_selection=all_v2_valid_rows`; their explicit `v2_row_filter_applied=false`,
`row_filter_policy=all_cached_detector_rows_no_v2_ablation` and contracts are
authoritative. The metadata wording was subsequently clarified in code without
altering numeric artifacts or overwriting the original experiment reports.

## 9. No-V2 control results — completed for both priors

| Prior / no V2 | Median TE cm | Mean TE cm | P90 TE cm | Median RE deg | Matcher mean ms |
|---|---:|---:|---:|---:|---:|
| Vanilla | 21.0699 | 30.7249 | 56.2980 | .269171 | 6.285 |
| MAtCHA | 10.2169 | 19.0043 | 48.6155 | .178072 | 5.680 |

Vanilla median TE is essentially unchanged versus F0 (21.1187), with modest
rotation/tail changes. MAtCHA median TE/RE degrade. Map size and matcher time
increase; F0 costs about 5.01/4.77 ms for matching. No-V2 does not materially
close the weak-prior gap and is not promoted. V2's clearest benefit in this
control is a smaller map; a universal accuracy improvement is not established.

The no-V2 weak/strong TE ratio is about 2.06, versus F0 2.18. This is not a
robustness win: part of the smaller ratio comes from worsening the strong
prior result. Together with the depth control, the comparison does not support
the anticipated ordering 'depth worst sensitivity, plain Track middle, F0 best'.
That negative result is retained, not reframed as confirmation.

## Decisions

| Hypothesis / concrete intervention | Evidence status | Decision |
|---|---|---|
| Current weak map has descriptor/matching recovery space | Supported by isolated fixed-XYZ real-descriptor diagnostic | Investigate observation/identity representation, not an RGB deployment claim |
| Internal geometry evidence identifies difficult subsets | Supported as association, not physical truth | Keep audits and unknown states |
| Hard surface-conflict filtering improves weak-prior poses | Not supported by full test | Keep disabled |
| Carrying geometric quality through descriptor bins closes gap | Both concrete variants fail | Keep disabled; no threshold sweep |
| One-pass bounded featuremetric correction improves final accuracy | Full eligible-target test fails | Keep disabled; not a theorem about all featuremetric methods |
| Existing-view geometry extension pilot improves final poses | Earlier pilot fails | Keep disabled; no claim about all possible new baselines |
| Direct depth or no-V2 removes prior dependence | Not supported | Retain as control baselines, not replacements |
| Fixed-ray reconstruction is independent of unused auxiliary depth | Regression test passes exactly | Claim only conditional representation decoupling |
| Fully correct physical identities / arbitrary-quality prior invariance | Insufficient reference / unsupported broad guarantee | Do not manufacture a positive or negative proof |

Default F0 is unchanged. No candidate here is promoted as a deployable accuracy
improvement. No offline self-localization feedback or dense online rendering is
enabled. Timing is recorded but this Base-protocol study does not certify 10 Hz.

All artifacts: `/mnt/pool/sqy/anygsloc_weak_prior_protocol_20260907/`, with arms
`OldHospital_vanilla_2dgs` and `OldHospital_matcha`. New map subdirectories:
`real_descriptor_diagnostic`, `geometry_descriptor_alignment/{within_bin,across_bin}`,
`depth_geometry_control`; real-track audit `nvm_geometry_reference.json`.
Each full pose evaluation is in the arm's `evaluation/` directory with its
named condition and `_seed2026` suffix. Original F0 maps remain unchanged.
