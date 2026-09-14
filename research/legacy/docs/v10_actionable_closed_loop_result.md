# V10 actionable self-localization feedback result

## Frozen contract and final decision

V10 starts and ends with the same accepted V2 pre-association rebuild M0:

- map SHA256: `711855ea46fdaede2e49a306cb56d59ae432a1568a881798c3223b2d36f108f3`;
- 164,871 Anchors and exactly one descriptor per Anchor;
- geometry, Tracks, triangulation and Anchor registry frozen;
- no LOO, test pose, test RGB, real training RGB, map-conditioned second pass,
  descriptor adapter, prototype expansion or Anchor addition.

All V10 candidates were rejected by their frozen gates.  The formal deployed
state therefore remains bit-exact M0 with the native SuperPoint frontend.  The
real test stays sealed.

## Closed-loop repair

The loop now has four explicit and separately auditable stages:

```text
novel-pose planner
  -> V2 render certificate (sample gate + per-row reliability)
  -> causal observer (diagnosis, not permission)
  -> bounded action + exact matching/PoseLib counterfactual
  -> independent safety
  -> one fresh confirmation
  -> ACCEPT or bit-exact ROLLBACK
```

The planner no longer interpolates mapping trajectories or uses unrestricted
look-at views.  It samples 0.9--2.0 baseline novel views, bounds parent-view
rotation to 40 degrees, targets current confusion pairs and makes pose families
unique.  It produced 48 safety and 40 disjoint confirmation poses.  Render V2
accepted 48/48 safety and 39/40 confirmation images.  Safety and confirmation
share no pose family and exclude all prior V9 feedback/confirmation families.

The certificate is deliberately neither a whole-image-only rejector nor a
pixel-mask-only mechanism.  REJECT/UNCERTAIN images cannot drive an action;
inside ACCEPT images, invalid/uncertain rows remain filtered or ignored.

## Map descriptor actions

Family-balanced feedback descriptors were admitted only when the candidate
Anchor was within Top-8 in at least three pose families, family dispersion was
at most 8 degrees median and 12 degrees P90, and a five-degree bounded spherical
update predicted at least two score flips in two families.  This reduced the
observer evidence to 33 candidate Anchors:

- 13,673 rejected for insufficient families;
- 1,097 rejected for inconsistent view-family descriptors;
- 36 rejected because a bounded update could not flip the confusion.

Exact per-Anchor matching plus standard PoseLib replay authorized only two
Anchors.  Their joint proposal changed zero of 48 independent safety poses, so
it was rolled back before fresh confirmation.

Because pose estimation is not an atomic one-correspondence task, V10 also
tested a theoretically cleaner causal unit: eight disjoint co-occurrence groups
of sizes 8, 8, 7, 4, 2, 2, 1 and 1.  Every group was replayed with exact global
Top-1 and standard PoseLib.  No group passed.  The best group improved 9/15
affected families and had positive cumulative gain `0.03987`, but worsened 40%
of them; the allowed maximum was 20%.  This rules out “the single-Anchor gate
was merely too atomic” as the complete explanation.

The evidence instead shows a representation conflict: one globally shared
Anchor descriptor cannot reliably satisfy the view-dependent feedback.  A
typical action changes only one or two correspondences per query, and its gains
change sign across independent view families.  Relaxing the consistency or
worsening gates would turn the controller into confirmation-tuned descriptor
copying, so it is forbidden.

## Feedback-trained query detector

After all map actions froze, the same observer was converted into an action
that matches what a detector can actually control.  At native SuperPoint
locations:

- positive means the deployed M0 global Top-1 reprojection error is at most
  4 px on a V2-valid row;
- negative means V2-invalid, or a confidently wrong Top-1 error of at least
  12 px;
- the 4--12 px margin and V2-uncertain rows are ignored.

This avoids the earlier conceptual error of labeling a keypoint positive merely
because a correct Anchor occurs somewhere in Top-64: a detector cannot repair
that descriptor ranking.  From 229 ACCEPT feedback renders, the family-disjoint
split contains 182 training and 47 validation views, with 108,753/143,727
positive/negative training cells.  Two fixed seeds trained on GPUs 1 and 2;
seed 2027 was selected by validation loss only (`0.70786`, separation `0.08849`).

The deployed candidate remains minimal:

```text
one frozen SuperPoint pass
  -> native corner score * learned reliability
  -> one NMS / Top-2048
  -> unchanged descriptors
  -> exact Full-M0 Top-1
  -> one standard PoseLib call
```

There is no query/map second pass, reranking, map-conditioned inference or
descriptor adaptation.

### Independent safety

| Frontend | Median TE | P90 TE | R5 | Catastrophic |
|---|---:|---:|---:|---:|
| Native | 1.601 cm | 23.415 cm | 83.33% | 3 |
| Feedback detector | 1.567 cm | 5.941 cm | 87.50% | 2 |

The paired median task gain was `0.00732`; 29/48 queries improved.  The frozen
safety gate passed.  A minority of severe regressions nevertheless made total
gain negative, so this risk signal was carried into confirmation rather than
discarded.

### One fresh confirmation

| Frontend | Median TE | P90 TE | R5 | Catastrophic |
|---|---:|---:|---:|---:|
| Native | 1.288 cm | 4.300 cm | 89.74% | 0 |
| Feedback detector | 1.412 cm | 4.504 cm | 92.31% | 0 |

The detector again gained one R5 success, but median and P90 translation both
regressed.  Its paired median task gain was only `0.000278`, below the frozen
`0.001` minimum, with 20/39 improvements.  Confirmation therefore returned
`ROLLBACK`.

## Conclusions

1. The repaired planner and V2 admission path work: unlike the old extreme
   look-at batch, the novel queries are predominantly localizable and provide
   useful feedback without reusing mapping trajectories.
2. The causal observer is useful as diagnosis.  Its precision-deficit rows are
   not automatically safe map-edit permissions.
3. Under the desired one-descriptor-per-Anchor representation, both atomic and
   bounded group descriptor reconstruction are too view-conflicted to write
   back safely.  Anchor deletion had already failed in V9; it is not reopened.
4. Feedback-trained query allocation is the strongest surviving direction.  It
   reproducibly improves R5/tail behavior, but its median effect is not stable
   enough for deployment.
5. The final accepted method remains V2 M0 plus native SuperPoint.  V10 improves
   the theoretical completeness and failure localization of the feedback loop,
   but honestly produces no accepted online action.  Test must not be opened to
   rescue or choose these rejected candidates.

The next scientifically admissible improvement is not another broader map
mutation.  It is a better detector target that predicts *pose contribution*
rather than per-point Top-1 correctness, trained on new render-only feedback
families and evaluated on a newly generated confirmation registry.  It should
retain the present single-pass frontend and all rollback gates.

## Main artifacts

- Contract: `configs/v10_actionable_anchor_feedback.yaml`
- Descriptor proposal:
  `/mnt/pool/sqy/lafgs_v10_actionable_20260828/StMarysChurch/descriptor_proposal/anchor_descriptor_proposal.pt`
- Single-action safety decision:
  `/mnt/pool/sqy/lafgs_v10_actionable_20260828/StMarysChurch/safety_decision.json`
- Group action audit:
  `/mnt/pool/sqy/lafgs_v10_actionable_20260828/StMarysChurch/group_descriptor_map/group_action_audit.pt`
- Novel query plans:
  `/mnt/pool/sqy/lafgs_v10_actionable_20260828/StMarysChurch/query_plans_v2`
- Feedback detector dataset:
  `/mnt/pool/sqy/lafgs_v10_actionable_20260828/StMarysChurch/feedback_detector_data`
- Selected detector checkpoint:
  `/mnt/pool/sqy/lafgs_v10_actionable_20260828/StMarysChurch/feedback_detector_seed2027.pt`
- Detector safety decision:
  `/mnt/pool/sqy/lafgs_v10_actionable_20260828/StMarysChurch/detector_safety_decision.json`
- Detector confirmation decision:
  `/mnt/pool/sqy/lafgs_v10_actionable_20260828/StMarysChurch/detector_confirmation_decision.json`
