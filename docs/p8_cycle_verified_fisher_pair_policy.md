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
- Three thin runners now preserve that boundary end to end:
  `materialize_cycle_verified_pair_probe.py` constructs and matches the frozen
  two-arm union once, `select_cycle_verified_fisher_pairs.py` consumes only the
  SHA-bound probe, and `compare_cycle_verified_fisher_mechanism.py` replays the
  nearest/P8 subsets on the same probe before checking Track evidence and exact
  precomputed-match lineage.
- Synthetic CPU tests cover hash tampering, aggregate-sidecar rejection,
  exact-budget closure, graph coverage, scale invariance, hard failure, and
  matcher bypass.  The CLI-level synthetic test additionally carries the two
  persisted artifacts into the normal Track builder while replacing the
  descriptor matcher with a forbidden-call sentinel.
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

The Stairs probe is materialized once with:

```bash
PYTHONPATH=. python -m scripts.materialize_cycle_verified_pair_probe \
  --scene stairs \
  --query-cache "$QUERY_CACHE" \
  --expected-query-cache-sha256 "$QUERY_CACHE_SHA" \
  --nearest-factor "$NEAREST_FACTOR" \
  --expected-nearest-factor-sha256 "$NEAREST_FACTOR_SHA" \
  --geometry-factor "$GEOMETRY_FACTOR" \
  --expected-geometry-factor-sha256 "$GEOMETRY_FACTOR_SHA" \
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

After Track construction consumes `probe_track_build_inputs(probe, selection)`,
the mechanism comparator requires explicit file and content hashes for both
artifacts plus exact control/variant factor and report hashes.  It rejects a
variant unless its factor attests both artifact lineages, exactly reproduces the
selected pair table, and records `track_pair_matches_reused=1` together with
`uses_precomputed_pair_matches=true`.

```bash
PYTHONPATH=. python -m scripts.compare_cycle_verified_fisher_mechanism \
  --scene stairs \
  --query-cache "$QUERY_CACHE" \
  --expected-query-cache-sha256 "$QUERY_CACHE_SHA" \
  --probe "$PROBE" --expected-probe-sha256 "$PROBE_FILE_SHA" \
  --expected-probe-content-sha256 "$PROBE_CONTENT_SHA" \
  --selection "$SELECTION" \
  --expected-selection-sha256 "$SELECTION_FILE_SHA" \
  --expected-selection-content-sha256 "$SELECTION_CONTENT_SHA" \
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
  --maximum-cycle-reprojection-error-px 2.0 \
  --output "$P8_ROOT/stairs/mechanism_gate.json"
```

Input, hash, schema, or lineage failures exit 1 without producing a valid gate.
A valid comparison that fails any preregistered scientific gate persists the
STOP JSON and exits 2.  Only a valid GO exits 0.

## Current blocker and next executable step

There is no scientific result for P8 yet.  The old selected-pair factors cannot
be converted into a valid probe because they do not retain candidate keypoint
correspondences.  The orchestration is now ready; the next executable step is
the single bounded Stairs probe above, followed by selection and a reuse-aware
Track build.  Only a Stairs Stage-A/B pass authorizes the identically frozen
GreatCourt probe.  No real probe, GPU job, or test evaluation was started in
this implementation task.
