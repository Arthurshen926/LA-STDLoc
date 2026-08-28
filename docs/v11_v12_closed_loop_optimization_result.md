# V11/V12 self-localization feedback optimization result

## Final decision

V11 and V12 are both rolled back.  The accepted state remains the V2 M0 map
and native SuperPoint frontend.  No test query, test pose, real training RGB,
LOO operation, map mutation or descriptor mutation was used.

The result is not “the detector did nothing.”  It isolates a sharper boundary:
render-only scene reliability can change the localization tail and median, but
the sign of its per-query PoseLib effect is not predictable reliably enough for
safe deployment under the current one-descriptor/one-shot localization
contract.

## V11: pose-contribution detector

V10 supervised the detector with per-point Top-1 correctness.  V11 replaces
uniform pixel classification with an analytic set-contribution proxy:

- correct Top-1 points are weighted by spatial-cell rarity, off-axis support
  and inverse-depth parallax;
- confidently wrong points are weighted by Top-1/Top-2 margin and gross
  reprojection error;
- 4--12 px ambiguity and V2-uncertain rows are ignored;
- no point is removed and re-solved, so this is not LOO.

The data remain 182 train and 47 family-disjoint render validation queries. Two
seeds were trained on GPUs 1 and 2 from the synthetic V2 clean-support detector;
seed 2028 was frozen by validation loss.

### Bounded action designs

Three increasingly safe allocation policies were evaluated only on validation:

1. bounded continuous fusion strengths;
2. protected Native cores of 1024, 1536 and 1792 rows;
3. the complete Native-2048 set plus 256 learned tail rows.

Strong actions improved validation median TE by roughly 0.09--0.15 cm, but every
action that did so lost one R5 success. Even preserving all Native rows and only
adding 256 rows changed standard PoseLib enough to lose that success. Therefore
the issue is not merely accidental deletion of a critical Native row.

### Query-level abstention

A fixed natural confidence boundary was then used:

```text
mean learned reliability on Native Top-2048 >= 0.5
    -> use learned allocation
otherwise
    -> exact Native allocation
```

It activates on 23/47 validation queries and changes neither validation R5 nor
catastrophic count while reducing median TE from 1.172 to 1.026 cm. This was the
only V11 candidate allowed to reach a newly generated safety set.

The new planner excluded all V9 and V10 plan families, used 0.9--2.0 mapping
baselines, bounded parent rotation to 40 degrees and generated disjoint 64/64
safety/confirmation plans. V2 accepted 62 safety renders, marked one uncertain
and rejected one.

On the 62 ACCEPT safety queries:

| | Native | Gated detector |
|---|---:|---:|
| Median TE | 1.789 cm | 1.653 cm |
| P90 TE | 4.782 cm | 4.802 cm |
| R5 | 90.32% | 90.32% |
| Catastrophic | 0 | 0 |

The aggregate median looks positive, but it is not the causal decision metric.
Only 28 queries activated the detector; among them the conditional median task
gain was `-0.001846`, the cumulative gain was `-0.28019`, and 53.57% worsened.
The global paired median gain is exactly zero because inactive queries are exact
Native. The frozen safety gate therefore returned `ROLLBACK`. Fresh
confirmation was never rendered or evaluated.

This is an important guard against a misleading conclusion: a lower aggregate
median after selective intervention does not prove that the intervention
helped the selected queries.

## V12: causal action gate

V12 tests whether feedback can predict detector actionability rather than using
a manually chosen confidence statistic. For all 182 training feedback queries,
the closed loop executes exact paired Native/detector Top-1 plus standard
PoseLib offline. A standardized seven-feature linear gate receives only
query-side, pre-matching statistics:

- reliability mean, P10, standard deviation and fraction above 0.5;
- Native score mean and standard deviation;
- Native/detector keypoint overlap.

The positive target is actual paired task gain at least 0.001 without converting
an R5 success into a failure. Online, this gate would still choose one feature
set before a single matcher and a single PoseLib call.

The gate found 86 positive actions among 182 training queries, but reached only
52.2% training classification accuracy. On the untouched 47-query render
validation set it activated 24 queries, obtained conditional median gain
`0.01341`, but worsened 37.5% and again lost one R5 success. Restricting the gate
to its highest-confidence 8--24 predictions could not reduce the validation
worsening fraction below 33.3%; the frozen maximum is 25%.

V12 consequently stops at validation. No additional safety or confirmation
result is consumed.

## Method conclusion

The repaired feedback loop is now theoretically complete in the important
sense that diagnosis, actionability, bounded action, safety and confirmation
are distinct:

```text
novel certified render
 -> causal matching/PoseLib observer
 -> action-specific evidence
 -> bounded proposal
 -> paired safety
 -> one fresh confirmation
 -> accept or exact rollback
```

What it does **not** guarantee is that an admissible beneficial action exists.
The accumulated evidence now rules out the following as safe improvements on
this scene and representation:

- Anchor deletion based on false-winner frequency;
- bounded one-descriptor reconstruction, individually or in confusion groups;
- shared low-rank metric adaptation;
- uniform scene-detector modulation;
- pose-proxy weighting alone;
- fixed confidence abstention;
- a linear causal gate from cheap query-only statistics.

The practical conclusion is that current feedback is informative enough to
diagnose failures but not rich enough to control them safely while preserving
one descriptor per Anchor and one-shot query localization. More render samples
or looser gates would not repair this representation/action mismatch.

Any next claimed gain must change one of the currently binding assumptions and
register it explicitly—for example a genuinely view-conditioned map
representation or non-test real-domain calibration. Until that authority and
evidence exist, V2 M0 + Native SuperPoint is the correct deployed method.

## Artifacts

- V11 contract: `configs/v11_pose_contribution_detector.yaml`
- V12 contract: `configs/v12_causal_scene_action_gate.yaml`
- V11 dataset:
  `/mnt/pool/sqy/lafgs_v11_pose_detector_20260828/StMarysChurch/detector_data`
- Selected V11 detector:
  `/mnt/pool/sqy/lafgs_v11_pose_detector_20260828/StMarysChurch/detector_seed2028.pt`
- V11 query plans:
  `/mnt/pool/sqy/lafgs_v11_pose_detector_20260828/StMarysChurch/query_plans`
- V11 safety decision:
  `/mnt/pool/sqy/lafgs_v11_pose_detector_20260828/StMarysChurch/safety_decision.json`
- V12 action gate:
  `/mnt/pool/sqy/lafgs_v11_pose_detector_20260828/StMarysChurch/scene_action_gate_v2.pt`
- V12 validation decision:
  `/mnt/pool/sqy/lafgs_v11_pose_detector_20260828/StMarysChurch/gate_validation_decision_v2.json`
