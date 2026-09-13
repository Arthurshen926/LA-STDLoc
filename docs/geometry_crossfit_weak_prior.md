# Weak-prior geometry evidence: first measured audit

## Scope and decision

The completed OldHospital same-code rebuild reproduced historical per-query
TE/RE for both priors. No additional reconstruction is required. F0 medians
remain MAtCHA 9.7066 cm / 0.17135 degrees and vanilla 21.1187 cm / 0.28133 degrees.

This change implements a **read-only mapping geometry audit**, not an accepted
localization improvement. It does not restore offline self-localization,
change query processing, use source RGB/test GT, or write into a deployed map.
No new final-pose accuracy result is claimed.

The objective is conditional robustness: recover useful landmarks despite
imperfect primitive geometry, when reliable multiview image information
remains. Similar localization accuracy for arbitrarily corrupted priors is
not identifiable from the frozen prior alone. Internally consistent shared
errors cannot be reliably recognized without independent evidence.

## Implemented evidence

`evidence/geometry_crossfit.py` splits a Track's mapping cameras along their
dominant spatial extent into two disjoint groups. Each group needs at least
three observations, P75 ray parallax of 1 degree, and a bounded condition
number. Each independently fitted ray intersection predicts the OTHER group's
image observations. The threshold is 2 px for both held-out medians.

- `internally_consistent`: sufficient group geometry and passing predictions;
  **not** certified real-world accuracy.
- `unstable`: sufficient group geometry but failing cross-group predictions
  or held-out cheirality.
- `unknown`: short Track or inadequate geometry in one group; never an
  automatic rejection.

The optional surface evidence preserves BOTH signed common depth discrepancy
and within-Track discrepancy MAD. It does not subtract away the common bias
and discard it. Render depth is only diagnostic, not a constraint on xyz or
a rejection vote. Missing depth produces unknown evidence, not failure.
Expected/median depth agreement, normal consistency and visibility/occlusion
reasoning are NOT implemented here; depth conflict does not identify which
geometry source is wrong.

## Measured panel

Both maps were sampled at 2,048 uniformly spaced Anchor rows, including
completion rows. These are not matched physical identities across priors.
At most 32 original observations per Anchor were selected deterministically.
The diagnostic uses unweighted ray LS, not the production robust-IRLS subset;
its failures can include sensitivity to outliers and group partitioning.

| Evidence | MAtCHA | Vanilla 2DGS |
|---|---:|---:|
| Internally consistent | 655 | 534 |
| Unstable | 335 | 401 |
| Unknown | 1,058 | 1,113 |
| Unstable, yet original median residual <=2 px | 286 | 334 |
| Base Track sample count, excluding completion | 1,936 | 1,941 |
| Base Tracks with <10 observations | 1,266 | 1,323 |

Among testable sampled rows, unstable fractions are 33.8% / 42.9%. This is
internal consistency evidence, not an estimate of false-Anchor prevalence or
an attribution of the real localization error gap. Most unknown rows are
short Tracks (987 / 1,008 rows have fewer than six sampled observations).
LOO10's floor(n*0.1) budget cannot adjust the 65.4% / 68.2% of sampled base
Tracks that have fewer than ten observations. This confirms a coverage gap
in that intervention, not that short Tracks dominate real localization error.

Median / P90 absolute within-Track common relative rendered-depth discrepancy:
MAtCHA 0.59% / 4.85%; vanilla 0.86% / 7.41%. These numbers compare two model
estimates; they are not depth errors against ground truth. Surface diagnostics
leave the crossfit classifications unchanged by construction.

Artifacts are under
`/mnt/pool/sqy/anygsloc_weak_prior_protocol_20260907/{OldHospital_matcha,OldHospital_vanilla_2dgs}/geometry_crossfit_surface_panel2048.json`.
Each file records source/map/cache hashes, row selection, policy, and per-row
measurements. Pixel-center conversion uses the same `PIXEL_CENTER_OFFSET` as
the observation provider, and source CSR lineage is checked before analysis.

## Next decision, not automatic promotion

1. Relate frozen geometry evidence to independently checked real
   correspondences and actual query Anchor usage. Do not use nearest GT
   projections as proof of physical identity. Evaluate whether unstable rows
   really contribute more pose error, with short/long and coverage controls.
2. If supported, target only problematic Tracks: obtain independent existing
   mapping observations or correct multiview measurements and retriangulate.
   Unknown short Tracks need evidence acquisition, not blanket removal.
3. Run one isolated modification on Kings and Old, protect strong-prior
   performance, and judge median TE/RE plus tails and latency. No test-tuned
   variant should be presented as independent generalization evidence.
4. Test relative prior sensitivity against fixed-descriptor geometry-source
   controls (ray geometry versus rendered-depth lifting). Hold strong-prior
   accuracy intact: shrinking the gap by degrading it is not robustness.

Do not promote a hard geometry gate solely from these mapping statistics.
The common-wrong-surface unit test intentionally passes internal consistency;
this documents the blind spot rather than hiding it behind a quality score.
