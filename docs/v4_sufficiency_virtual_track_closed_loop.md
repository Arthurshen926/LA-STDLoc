# Sufficiency virtual-render Track closed loop

This is a bounded experimental entry point; the default V4 pipeline is
unchanged.  The frozen planner supplies virtual poses, and the experiment runs
Gaussian RGB/alpha/depth rendering, the standard SuperPoint frontend,
family-aware Track construction, pure-ray triangulation, and the shared
`UnifiedAnchorConstructor`.

The Top-8 dry-run gate was fixed before rendering.  It consumes only mapping
evidence: detector support, Track/Anchor counts, family independence, and view
bins. `gt_visible_diagnostic` remains null.  Top-32 refuses to run without a
passing Top-8 decision bound to the exact plan SHA.  Test localization is a
separate downstream step after the experimental map artifact is frozen.

The reported mapping pose solve is explicitly an observation oracle, not a
claimed leave-one-out generalization measurement: the query's observation may
participate in its Track geometry.  It is retained only to expose broken
geometry/PnP conventions during the closed-loop dry run.
