# P8 mapping-only `cycle_verified_fisher` pair-policy preregistration

## Decision and causal boundary

The next pair-policy objective is **verified triangle closure × dimensionless
bearing-Fisher information**, under an exact global pair budget and hard camera
graph coverage constraints.  It is not another overlap/parallax weight sweep.

The existing `lafgs_mapping_track_pair_sidecar` is insufficient for selecting
this policy.  Its `cycle_supported_edge_count` is produced only after the old
policy has selected pairs, descriptor/epipolar matching has run, and Tracks have
been assembled.  It retains aggregate counts and final Track indices, but not
the candidate-pool keypoint correspondences needed to ask whether an unselected
edge closes an exact descriptor triangle.  Using that sidecar as a selector
input would reverse cause and effect.

Consequently P8 requires one bounded mapping-only match probe.  The probe is an
explicit, SHA-bound artifact, and its selected correspondence subset is reused
directly by Track construction.  The selected pairs are **not matched a second
time**.  No test query, deployment metric, Map identity, or pose result is used
by either the probe or selector.

Implementation status at preregistration:

- `evidence/cycle_verified_fisher.py` implements the bounded candidate-union
  contract, exact correspondence probe schema/validator, exact three-keypoint
  cycle verification, scale-normalized bearing-Fisher utility, hard spanning
  forest/degree constraints, exact-budget selection, and probe-subset reuse.
- `evidence/triangulation.py` has a fail-closed precomputed-pair path so the
  selected probe rows enter the normal cycle/Track builder without another
  descriptor matcher call.
- Six thin runners now preserve that boundary end to end.  The proposal
  attestor binds only the archived pair tables to the fresh cache; the probe
  matches their union once; selection and the independent Stage-A comparator
  consume that same probe; the reuse-only Track runner materializes both the
  nearest control and P8 variant with a forbidden matcher-call sentinel; and
  the independent Stage-B comparator checks their exact shared lineage before
  applying Track gates.
- Synthetic CPU tests cover hash tampering, aggregate-sidecar rejection,
  exact-budget closure, graph coverage, scale invariance, hard failure, and
  matcher bypass.  The CLI-level synthetic test additionally carries the
  proposal, probe, selection and Stage-A gate through both normal Track builds
  while replacing the descriptor matcher with a forbidden-call sentinel.
- No real probe, GPU factor, fullchain, mapping pose, or formal test has run.

## Why this objective follows from the two-scene postmortem

All values below were computed read-only from the frozen mapping-only Stairs and
GreatCourt pair factors.  They are diagnosis for preregistration, not new test
evidence.  Exact factor/report SHA-256 provenance is recorded in the
machine-readable preregistration.

| topology / evidence diagnostic | Stairs nearest | Stairs parallax | GreatCourt nearest | GreatCourt parallax |
|---|---:|---:|---:|---:|
| exact pair budget | 7,450 | 7,450 | 5,254 | 5,254 |
| selected camera-graph components | 4 | **2** | 1 | **4** |
| selected-graph isolated cameras | 0 | 0 | 0 | **2** |
| camera triangles | 578 | **709** | 5,952 | **969** |
| pairs with at least one exact cycle-supported match | 1,068 | **1,728** | 5,044 | **1,653** |
| cycle-positive graph components | 1,527 | **1,034** | 6 | **773** |
| cycle-positive isolated cameras | 1,468 | **929** | 4 | **688** |
| cycle-supported correspondence edges | 76,786 | 49,832 | 1,938,336 | **338,293** |
| triangulated Tracks | 17,798 | **31,660** | 34,150 | **32,293** |
| triangulated covariance p90 (m2) | 0.06864 | **0.03524** | 28.14965 | **47.28497** |

GreatCourt therefore supplies the clean falsification: parallax below one degree
fell by 61.85 percentage points, but camera triangles fell by 83.72%, exact
cycle-supported correspondences fell by 82.55%, the selected graph disconnected,
and covariance p90 worsened by 67.98%.  Parallax was changed successfully while
usable identity geometry became less reliable.

Stairs moved in the opposite **topological** direction: camera triangles grew
22.66%, cycle-positive pair coverage grew 61.80%, components contracted, and
triangulated yield/conditioning improved.  Yet its total cycle-supported
correspondence count fell 35.10%.  Thus raw cycle count is not the target either:
concentrated redundant cycles may be less useful than broader, independently
conditioned closures.  This is why P8 combines exact closure with information
conditioning and keeps connectivity lexicographically hard.

## Fixed candidate universe and bounded probe

P8 V1 freezes the proposal universe before matching as the union of:

1. the frozen nearest-policy pair set; and
2. the equally budgeted frozen mapping-geometry proposal pair set.

The second arm is used only to bound which pairs receive one correspondence
probe.  Its overlap/parallax values never enter P8 utility or tie-breaking.  The
union is capped at `2 * frozen_pair_budget`; it must cover every mapping camera.
The already frozen pair indices imply these exact contracts:

| scene | selected budget | candidate union | ratio | union components | isolated | min degree |
|---|---:|---:|---:|---:|---:|---:|
| Stairs K1024/NMS4 | 7,450 | 14,835 | 1.9913x | 2 | 0 | 9 |
| GreatCourt K2048/NMS4 | 5,254 | 9,875 | 1.8795x | 1 | 0 | 5 |

The probe therefore costs less than two normal selected-pair matching passes,
not the 48-neighbour all-proposal search.  Because selected matches are reused,
total matching is the union size rather than `union + selected budget`.

The archived Stairs/GreatCourt factors predate the current K/NMS/cache-lineage
contract.  They are therefore **not** accepted as complete Track factors for
P8.  `attest_cycle_verified_pair_proposals.py` reads only their sorted pair
tables, records every unavailable lineage field, and produces a new pair-only
proposal attestation whose authoritative query order, K and NMS come from the
fresh cache.  No old match, Track, triangulation or geometry value is promoted.

The probe contract binds:

- mapping-only attestation, ordered query-name SHA, query-cache SHA, K and NMS;
- exact candidate-pair indices, keypoint counts, construction name/parameters;
- every matcher threshold, epipolar mode, and detector-score weighting;
- flattened reciprocal one-to-one source/target indices and confidence;
- a content SHA over all scientific fields and per-pair matcher diagnostics.

The selection artifact has its own content SHA and revalidates its exact budget,
candidate subset, graph diagnostics, and probe/candidate hashes before exposing
precomputed matches to Track construction.

Any old aggregate sidecar, missing candidate pair, stale hash, out-of-range
keypoint, duplicate reciprocal row, mismatched K/NMS, or test-query attestation
fails before selection.

## Objective

For a candidate camera triple `(i,j,k)`, a closure exists only when the three
probed reciprocal matches map exactly the same keypoint identity around all
three edges.  The three known mapping rays are intersected, positive depth is
required, and maximum three-view reprojection error must not exceed the frozen
`2.0 px` threshold.

For verified landmark `m`, with world bearing `r_q`, camera center `C_q`, and
intersected point `X_m`, P8 uses the bearing information matrix

```text
F_m = sum_q (I - r_q r_q^T) / ||X_m - C_q||^2
```

and utility

```text
u_m = geometric_mean(edge_confidence) * logdet(I + s^2 F_m),
```

where `s` is the median positive candidate-pool baseline.  `s^2 F_m` makes the
score invariant to a global world-unit rescaling; a synthetic 1x/10x test locks
that property.  This is a bearing-information objective, not a claim that the
current rendered depth covariance is an empirical localization posterior.

Selection is deterministic and lexicographic:

1. build a maximum-utility spanning forest that preserves every connected
   component of the candidate universe;
2. satisfy the preregistered per-camera degree floor, failing if the exact budget
   cannot do so;
3. greedily add the missing edge bundles that complete verified triangles by
   confidence-weighted Fisher gain per spent edge;
4. fill any remaining exact-budget slots by verified cycle/Fisher edge utility.

The final graph must have exactly the candidate-universe component count, zero
isolated mapping cameras, the degree floor, and exactly the frozen pair budget.
Disconnected trajectories are not joined by invented edges, while an arm may
not fragment a component that the candidate universe can connect.

## Preregistered two-scene gates

The machine-readable contract is
`docs/evidence/p8_cycle_verified_fisher_preregistration.json`.  All comparisons
use the same fresh query cache/K/NMS/query order and the frozen nearest control.
No threshold may be changed after either scene's probe is inspected.

### Stage A: contract and selector (both scenes)

- exact candidate-union count: Stairs 14,835; GreatCourt 9,875;
- probe pair count no greater than `2 * frozen budget`, content/lineage valid,
  and `uses_test_queries=false`;
- selected budgets exactly 7,450 / 5,254;
- selected component counts exactly 2 / 1, matching the candidate universes;
- zero selected-graph isolated cameras and minimum degree at least one;
- selected confidence-weighted verified-Fisher utility at least 1.05x the
  nearest subset evaluated from the **same** probe;
- completed verified keypoint triangles at least 0.98x nearest;
- fraction of mapping cameras participating in a completed verified triangle
  no lower than nearest (this is reported separately from raw graph coverage).

Failure stops before Track/fullchain.  In particular, a higher Fisher score may
not compensate for a disconnected graph or lost closure coverage.

### Stage B: Track mechanism (both scenes)

- triangulated Tracks at least 0.98x nearest;
- broad eligible Tracks at least 0.98x nearest;
- high-confidence Tracks at least 0.98x nearest;
- triangulated covariance p90 no greater than 1.05x nearest;
- mapping-query fraction with broad Track support no lower than nearest;
- exact selected probe rows are reused, and the standard cycle/Track builder
  reports `track_pair_matches_reused=1`.

Both Stairs and GreatCourt must pass.  A Stairs-only Go remains scene-specific;
a GreatCourt failure stops P8 before any fullchain or pose run.
Accordingly, the per-scene Stage-B report can only emit
`scene_specific_mechanism_pass` plus `requires_other_scene=true`; it never
contains a fullchain authorization.  A separate cross-scene aggregator is
required.  The CPU-only aggregator is implemented independently of either
scene run; it accepts only the two explicit Stage-B gate paths and their expected
SHA-256 values, and cannot turn one scene into a two-domain authorization.

### Stage C: fullchain and mapping pose

Only after both mechanism gates pass, rebuild each scene's canonical Map,
function graph, provenance, teacher, compact artifacts, and metric from the P8
Track factor.  Use the existing locked q256 x seeds 2026/2027/2028 mapping-pose
gate unchanged: every seed must satisfy its no-regression checks and the
three-seed mean must have a preregistered substantive improvement.  Then run the
existing `office2_5b` tail no-regression sentinel.  Formal test remains forbidden
until all mapping-only gates pass and the method/default decision is frozen.

## Thin CLI execution contract

All scientific axes and all serialized inputs are explicit.  `--scene stairs`
requires K1024/NMS4/budget7450/union14835/components2; `--scene greatcourt`
requires K2048/NMS4/budget5254/union9875/components1.  A mismatch fails before
matching.  The matcher thresholds are also mandatory arguments rather than
hidden defaults, and V1 rejects values other than
`0.65/0.01/2.0px/topK1/-1.0/-1.0`.

First attest the archived inputs as **pair proposals only**.  The printed file
and content hashes become mandatory downstream inputs:

```bash
PYTHONPATH=. python -m scripts.attest_cycle_verified_pair_proposals \
  --scene stairs \
  --query-cache "$QUERY_CACHE" \
  --expected-query-cache-sha256 "$QUERY_CACHE_SHA" \
  --mapping-scope-equivalence "$MAPPING_SCOPE_EQUIVALENCE" \
  --expected-mapping-scope-equivalence-sha256 "$MAPPING_SCOPE_EQUIVALENCE_SHA" \
  --nearest-source "$ARCHIVED_NEAREST_FACTOR" \
  --expected-nearest-source-sha256 "$ARCHIVED_NEAREST_FACTOR_SHA" \
  --geometry-source "$ARCHIVED_GEOMETRY_FACTOR" \
  --expected-geometry-source-sha256 "$ARCHIVED_GEOMETRY_FACTOR_SHA" \
  --expected-query-names-sha256 "$QUERY_NAMES_SHA" \
  --expected-mapping-keypoints 1024 \
  --expected-nms-radius 4 \
  --expected-pair-budget 7450 \
  --expected-candidate-pair-count 14835 \
  --expected-candidate-components 2 \
  --output "$P8_ROOT/stairs/pair_proposals.pt"
```

Then materialize the one bounded probe from that attestation.  The archived
factors are no longer factor-lineage inputs to this command:

```bash
PYTHONPATH=. python -m scripts.materialize_cycle_verified_pair_probe \
  --scene stairs \
  --query-cache "$QUERY_CACHE" \
  --expected-query-cache-sha256 "$QUERY_CACHE_SHA" \
  --mapping-scope-equivalence "$MAPPING_SCOPE_EQUIVALENCE" \
  --expected-mapping-scope-equivalence-sha256 "$MAPPING_SCOPE_EQUIVALENCE_SHA" \
  --proposals "$P8_ROOT/stairs/pair_proposals.pt" \
  --expected-proposals-sha256 "$PROPOSALS_FILE_SHA" \
  --expected-proposals-content-sha256 "$PROPOSALS_CONTENT_SHA" \
  --expected-query-names-sha256 "$QUERY_NAMES_SHA" \
  --expected-mapping-keypoints 1024 \
  --expected-nms-radius 4 \
  --expected-pair-budget 7450 \
  --expected-candidate-pair-count 14835 \
  --expected-candidate-components 2 \
  --minimum-similarity 0.65 \
  --minimum-margin 0.01 \
  --maximum-epipolar-error-px 2.0 \
  --epipolar-candidate-topk 1 \
  --epipolar-recovered-minimum-similarity -1.0 \
  --epipolar-recovered-minimum-margin -1.0 \
  --device cuda \
  --output "$P8_ROOT/stairs/pair_match_probe.pt"
```

Its printed file/content hashes become mandatory selector inputs:

```bash
PYTHONPATH=. python -m scripts.select_cycle_verified_fisher_pairs \
  --scene stairs \
  --query-cache "$QUERY_CACHE" \
  --expected-query-cache-sha256 "$QUERY_CACHE_SHA" \
  --mapping-scope-equivalence "$MAPPING_SCOPE_EQUIVALENCE" \
  --expected-mapping-scope-equivalence-sha256 "$MAPPING_SCOPE_EQUIVALENCE_SHA" \
  --probe "$P8_ROOT/stairs/pair_match_probe.pt" \
  --expected-probe-sha256 "$PROBE_FILE_SHA" \
  --expected-probe-content-sha256 "$PROBE_CONTENT_SHA" \
  --expected-query-names-sha256 "$QUERY_NAMES_SHA" \
  --expected-mapping-keypoints 1024 \
  --expected-nms-radius 4 \
  --expected-pair-budget 7450 \
  --expected-candidate-pair-count 14835 \
  --expected-candidate-components 2 \
  --minimum-camera-degree 1 \
  --maximum-cycle-reprojection-error-px 2.0 \
  --output "$P8_ROOT/stairs/pair_selection.pt"
```

Before any Track construction, independently compare the nearest proposal arm
and the selected arm from that exact probe.  A valid scientific STOP is written
and exits 2; only a valid GO authorizes the reuse-only runner:

```bash
PYTHONPATH=. python -m scripts.compare_cycle_verified_fisher_stage_a \
  --scene stairs \
  --query-cache "$QUERY_CACHE" \
  --expected-query-cache-sha256 "$QUERY_CACHE_SHA" \
  --mapping-scope-equivalence "$MAPPING_SCOPE_EQUIVALENCE" \
  --expected-mapping-scope-equivalence-sha256 "$MAPPING_SCOPE_EQUIVALENCE_SHA" \
  --proposals "$P8_ROOT/stairs/pair_proposals.pt" \
  --expected-proposals-sha256 "$PROPOSALS_FILE_SHA" \
  --expected-proposals-content-sha256 "$PROPOSALS_CONTENT_SHA" \
  --probe "$P8_ROOT/stairs/pair_match_probe.pt" \
  --expected-probe-sha256 "$PROBE_FILE_SHA" \
  --expected-probe-content-sha256 "$PROBE_CONTENT_SHA" \
  --selection "$P8_ROOT/stairs/pair_selection.pt" \
  --expected-selection-sha256 "$SELECTION_FILE_SHA" \
  --expected-selection-content-sha256 "$SELECTION_CONTENT_SHA" \
  --expected-query-names-sha256 "$QUERY_NAMES_SHA" \
  --expected-mapping-keypoints 1024 \
  --expected-nms-radius 4 \
  --expected-pair-budget 7450 \
  --expected-candidate-pair-count 14835 \
  --expected-candidate-components 2 \
  --maximum-cycle-reprojection-error-px 2.0 \
  --output "$P8_ROOT/stairs/stage_a_gate.json"
```

On Stage-A GO, run the same reuse-only Track materializer twice.  It calls
`probe_pair_subset_track_build_inputs` for the nearest control and
`probe_track_build_inputs` for the variant; a sentinel makes any descriptor
matcher re-entry a hard error.  The manifest and frozen Track payload are the
same fresh-cache-bound Stage-A inputs used by the existing Track factor runner.

```bash
for ARM in nearest_control variant; do
  PYTHONPATH=. python -m scripts.materialize_cycle_verified_track_factor \
    --scene stairs \
    --arm "$ARM" \
    --manifest "$STAGE_A_MANIFEST" \
    --expected-manifest-sha256 "$STAGE_A_MANIFEST_SHA" \
    --frozen-track-payload "$FROZEN_TRACK_PAYLOAD" \
    --expected-frozen-track-payload-sha256 "$FROZEN_TRACK_PAYLOAD_SHA" \
    --query-cache "$QUERY_CACHE" \
    --expected-query-cache-sha256 "$QUERY_CACHE_SHA" \
    --mapping-scope-equivalence "$MAPPING_SCOPE_EQUIVALENCE" \
    --expected-mapping-scope-equivalence-sha256 "$MAPPING_SCOPE_EQUIVALENCE_SHA" \
    --proposals "$P8_ROOT/stairs/pair_proposals.pt" \
    --expected-proposals-sha256 "$PROPOSALS_FILE_SHA" \
    --expected-proposals-content-sha256 "$PROPOSALS_CONTENT_SHA" \
    --probe "$P8_ROOT/stairs/pair_match_probe.pt" \
    --expected-probe-sha256 "$PROBE_FILE_SHA" \
    --expected-probe-content-sha256 "$PROBE_CONTENT_SHA" \
    --selection "$P8_ROOT/stairs/pair_selection.pt" \
    --expected-selection-sha256 "$SELECTION_FILE_SHA" \
    --expected-selection-content-sha256 "$SELECTION_CONTENT_SHA" \
    --stage-a-gate "$P8_ROOT/stairs/stage_a_gate.json" \
    --expected-stage-a-gate-sha256 "$STAGE_A_GATE_SHA" \
    --expected-query-names-sha256 "$QUERY_NAMES_SHA" \
    --expected-mapping-keypoints 1024 \
    --expected-nms-radius 4 \
    --expected-pair-budget 7450 \
    --expected-candidate-pair-count 14835 \
    --expected-candidate-components 2 \
    --device cuda \
    --output-dir "$P8_ROOT/stairs/track_$ARM"
done
```

Finally, Stage B requires exact control/variant factor and report hashes.  It
independently rejects either arm unless both bind the same proposal, probe,
selection, Stage-A gate and matcher; both must reproduce their exact pair subset
with `track_pair_matches_reused=1` and
`uses_precomputed_pair_matches=true`.

```bash
PYTHONPATH=. python -m scripts.compare_cycle_verified_fisher_mechanism \
  --scene stairs \
  --query-cache "$QUERY_CACHE" \
  --expected-query-cache-sha256 "$QUERY_CACHE_SHA" \
  --mapping-scope-equivalence "$MAPPING_SCOPE_EQUIVALENCE" \
  --expected-mapping-scope-equivalence-sha256 "$MAPPING_SCOPE_EQUIVALENCE_SHA" \
  --proposals "$P8_ROOT/stairs/pair_proposals.pt" \
  --expected-proposals-sha256 "$PROPOSALS_FILE_SHA" \
  --expected-proposals-content-sha256 "$PROPOSALS_CONTENT_SHA" \
  --probe "$P8_ROOT/stairs/pair_match_probe.pt" \
  --expected-probe-sha256 "$PROBE_FILE_SHA" \
  --expected-probe-content-sha256 "$PROBE_CONTENT_SHA" \
  --selection "$P8_ROOT/stairs/pair_selection.pt" \
  --expected-selection-sha256 "$SELECTION_FILE_SHA" \
  --expected-selection-content-sha256 "$SELECTION_CONTENT_SHA" \
  --stage-a-gate "$P8_ROOT/stairs/stage_a_gate.json" \
  --expected-stage-a-gate-sha256 "$STAGE_A_GATE_SHA" \
  --control-factor "$CONTROL_FACTOR" \
  --expected-control-factor-sha256 "$CONTROL_FACTOR_SHA" \
  --control-report "$CONTROL_REPORT" \
  --expected-control-report-sha256 "$CONTROL_REPORT_SHA" \
  --variant-factor "$VARIANT_FACTOR" \
  --expected-variant-factor-sha256 "$VARIANT_FACTOR_SHA" \
  --variant-report "$VARIANT_REPORT" \
  --expected-variant-report-sha256 "$VARIANT_REPORT_SHA" \
  --expected-query-names-sha256 "$QUERY_NAMES_SHA" \
  --expected-mapping-keypoints 1024 --expected-nms-radius 4 \
  --expected-pair-budget 7450 --expected-candidate-pair-count 14835 \
  --expected-candidate-components 2 \
  --output "$P8_ROOT/stairs/stage_b_gate.json"
```

Every P8 entry point requires the cache itself to contain
`uses_test_queries=false`, or the caller must provide the same explicit SHA-bound
mapping-only V2 sparse-refresh equivalence shown above.  A missing cache flag is
never interpreted as `false`, and the proof is propagated into proposal, gate,
and Track lineage.

Input, hash, schema, scope, or lineage failures exit 1 without producing a valid
gate.  A valid comparison that fails any preregistered scientific gate persists
the STOP JSON and exits 2.  A valid per-scene pass exits 0 but only emits
`SCENE_PASS_REQUIRES_OTHER_SCENE`; it cannot authorize fullchain by itself.

After both independently produced Stage-B files exist, aggregate them without
loading a GPU or supplying any additional scientific axes:

```bash
PYTHONPATH=. python -m scripts.aggregate_cycle_verified_fisher_cross_scene \
  --stairs-stage-b-gate "$P8_ROOT/stairs/stage_b_gate.json" \
  --expected-stairs-stage-b-gate-sha256 "$STAIRS_STAGE_B_SHA" \
  --greatcourt-stage-b-gate "$P8_ROOT/greatcourt/stage_b_gate.json" \
  --expected-greatcourt-stage-b-gate-sha256 "$GREATCOURT_STAGE_B_SHA" \
  --output "$P8_ROOT/cross_scene_stage_b_gate.json"
```

The aggregator recursively rehashes each gate's Stage-A, proposal/probe/
selection, Track factor/report, manifest, frozen Track and mapping-scope inputs.
It requires the exact Stairs and GreatCourt scene contracts, a 9/9 Stage-A pass,
internally consistent Stage-B booleans, same-probe reuse lineage, and one shared
compiled policy identity after removing only the preregistered scene-specific
K/budget/calibration fields.  Two copies of one scene, a test-tainted input, a
hash/lineage mismatch, or a policy-identity mismatch exit 1 without output.  A
valid scientific failure in either scene writes `STOP_BEFORE_FULLCHAIN` and exits
2.  Only two valid scene passes write `GO_TO_FULLCHAIN_MAPPING_POSE`; even that
mapping-only Go records `authorizes_test=false`.

## Current Stairs result and next executable step

The real Stairs mapping-only sequence is now complete.  Its proposal, bounded
probe, exact-budget selection, independent Stage A, two reuse-only Track factors,
and independent Stage B all validate against the preregistered contract.
Stage A passes 9/9 gates: Fisher utility increases 64.03659 -> 638.61841,
completed verified triangles increase 84,878 -> 825,469, and participating
camera fraction increases 0.2660 -> 0.8665.  Stage B passes 8/8 gates:
triangulated/broad/high-confidence Tracks increase 15,053/14,276/41 ->
17,384/16,634/51, while triangulated covariance p90 decreases
0.05501186 -> 0.03653227 m2 and broad-query coverage stays 1.0.

The hash-bound Stage-B decision is `SCENE_PASS_REQUIRES_OTHER_SCENE`, not a
fullchain authorization.  The only authorized next execution is the identically
frozen GreatCourt Stage-A/B sequence.  GreatCourt has not run; neither scene has
run a P8 fullchain, mapping pose, formal test, or default-method switch.

The first real selector execution was separately recorded as invalid after
1,592.086 seconds without an output; it exposed an engineering complexity
blocker, not a scientific Stop.  The exact incremental selector at commit
`199c187acd8a6df018e3630fe0babda3739e68c1` completed the same formal selection
in about 188 seconds.  Runtime is not a scientific gate.  The complete result,
artifact hashes, gate values, and control/variant lineage diff are in
`docs/p8_cycle_verified_fisher_stairs_result.md` and
`docs/evidence/p8_cycle_verified_fisher_stairs_result.json`.
