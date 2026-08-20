# V4 hard-panel Anchor failure-chain audit

This extends the diagnostic-only L1--L5 audit to the existing frozen hard
scene panel. No map, threshold, metric, descriptor, selector, or PoseLib
parameter was changed.

## Frozen subset

For each scene, the audit selects the four mapping-LOO queries with the largest
translation error in the already-produced formal Top-1 statistics. This is a
bounded `tail4` diagnostic, not a full-scene failure-category distribution.
The selection uses no test query and does not feed any result back into map
construction or hyperparameter choice.

All seven requested scenes had the exact materialized map, identity metric,
support-repaired Track payload, positive teacher, appearance cache, and scene
calibration referenced by their frozen Top-1 report. Therefore no scene was
blocked. Large query caches were loaded one scene at a time.

## Results

`R@K` is the detector-accessible-row-weighted correct-Anchor recall across the
four audited queries. `Geom` is the number with L1 PnP-solvable visible
geometry; `Oracle` is the number for which direct standard PoseLib on the
maximum GT-positive matching is correct at 5 cm / 5 degrees.

| Scene | Frozen formal failures | >=100 cm | L1/L2/L3/L4/L5 | Eligible rows | R@1/2/4/8/16/32 (%) | Geom | Oracle |
|---|---:|---:|---:|---:|---|---:|---:|
| GreatCourt | 219 | 21 | 0/2/1/1/0 | 269 | 4.83/4.83/5.95/7.81/9.29/10.41 | 4/4 | 2/4 |
| StMarysChurch | 132 | 103 | 0/0/2/2/0 | 353 | 1.98/2.55/3.97/5.95/8.78/11.90 | 4/4 | 4/4 |
| office2_5a | 259 | 226 | 3/0/1/0/0 | 9 | 0/0/0/0/0/0 | 1/4 | 1/4 |
| office2_5b | 395 | 255 | 0/0/3/1/0 | 101 | 0/0/0.99/1.98/3.96/4.95 | 4/4 | 4/4 |
| office1_gates381 | 189 | 132 | 1/0/1/2/0 | 93 | 4.30/5.38/7.53/9.68/9.68/11.83 | 3/4 | 3/4 |
| office1_manolis | 161 | 156 | 1/1/1/1/0 | 16 | 50.00/62.50/62.50/62.50/62.50/68.75 | 3/4 | 2/4 |
| apt2_luke | 97 | 90 | 1/2/0/1/0 | 13 | 30.77/38.46/38.46/38.46/38.46/38.46 | 3/4 | 1/4 |

The formal-failure column counts mapping queries that fail 5 cm / 5 degrees in
the frozen full Top-1 report. It is context only; only tail4 receives L1--L5
labels. The high recall percentages for manolis and apt2_luke have very small
eligible denominators and must not be compared directly with the Cambridge
rows.

Across all 28 audited queries, the ordered failure counts are L1=6, L2=5,
L3=9, L4=8, and L5=0. No tail is first attributed to an unchanged PoseLib
solver gap.

## Interpretation

- `office2_5a` is coverage/geometry dominated in this extreme tail: three of
  four queries lack PnP-solvable visible Anchor geometry. Descriptor changes
  cannot recover those three cases.
- `office2_5b` is descriptor dominated: all four have solvable geometry and a
  correct direct oracle pose, but three fail before four correct Top-32
  correspondences survive.
- GreatCourt combines detector access, descriptor recall, and candidate
  structure. Its two L2 tails have solvable visible geometry but fewer than
  four detector-accessible GT-positive correspondences.
- StMarysChurch has correct oracle poses for all four tails; two are descriptor
  failures and two retain enough Top-32 positives but fail candidate
  organization.
- gates381 and manolis are genuinely mixed-mechanism scenes. A single global
  matching or geometry patch cannot explain all four extreme failures.
- apt2_luke is primarily detector-access limited in this subset, with one L1
  and one L4 case.

The evidence manifest records the full JSON and tensor-sidecar SHA-256 values:
`docs/evidence/v4_query_anchor_failure_chain_hard_panel_tail4.json`.

