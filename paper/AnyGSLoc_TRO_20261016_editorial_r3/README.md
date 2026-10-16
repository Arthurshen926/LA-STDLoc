# AnyGSLoc — IEEE T-RO Editorial Rewrite r3

This directory is the active editorial working copy for the AnyGSLoc IEEE Transactions on Robotics submission.

## Source and isolation

- Previous manuscript: `paper/AnyGSLoc_TRO_20261016_method_narrative_r2/`
- Active copy: `paper/AnyGSLoc_TRO_20261016_editorial_r3/`
- Earlier manuscript directories are not modified by this round.
- Scientific results, figures, tables, and algorithm implementations are treated as frozen unless a later review establishes a factual or consistency error.

## Editorial objective

The target is not merely grammatical polish. The manuscript is being rewritten as an insight-driven T-RO paper with a clear scientific chain:

**Gaussian scene as rendering representation → rendered observations → correspondence tracks → track-grounded localization landmarks → localization-aware compact map → direct renderer-free localization.**

The central distinction is deliberately simple: a Gaussian primitive is a rendering unit, whereas a localization landmark must explain the image correspondence used for pose estimation. Gaussian rendering supplies offline observations and geometric cues; the matched pixels and calibrated multiview geometry determine the exported landmark.

The writing follows the local IEEE writing-support guidance in `writing_support/ieee-skills/`, while the scientific record remains authoritative.

## Current round

Round 01–02 rewrote:
- `sections/01_introduction.tex`
- `sections/02_related_work.tex`
- `sections/03_method.tex`
- `sections/04_experiments.tex`
- `sections/05_conclusion.tex`
- the abstract in `main.tex`

The method now uses:
1. Problem Formulation and Map Representation;
2. Track Construction from Rendered Observations;
3. Track-Based Landmark Reconstruction;
4. Localization-Aware Landmark Selection;
5. Single-Stage Query Localization.

Observation filtering is kept under that exact name. Selection is presented as quality pruning → coverage completion → pose-aware completion, with redundancy cleanup folded into the last stage.

## Validation

The manuscript currently builds successfully with:

```
bash paper/AnyGSLoc_TRO_20261016_editorial_r3/build.sh
```

The current build produces a 15-page IEEEtran article including references. There is no appendix in the main article.

A legacy validation script in this copied tree still contains assumptions about earlier wording (notably the query-mask sentence); it should not be treated as the authoritative editorial check until it is updated to the current writing policy. The successful LaTeX build is current.

## Next editorial pass

The next pass should be a claim-evidence audit rather than another blind rewrite:
- trace every substantive Introduction claim into Method and Experiments;
- verify Related Work descriptions against the cited papers;
- check that the three contributions are non-overlapping and each has direct experimental evidence;
- inspect Method equations for symbol lifecycle, geometric assumptions, and exact correspondence to implementation;
- tighten remaining implementation prose only where it obscures the scientific idea;
- inspect the PDF page-by-page for paragraph rhythm, figure placement, table density, and awkward page breaks.

No experiment or algorithm change is intended as part of this writing workflow.
