# 7Scenes Track repair selective port

The archived branch `codex/v4-7scenes-track-repair` is not merged wholesale.
Its pipeline, calibration, cache, Track sidecar, and factor-runner changes have
all been superseded by stricter mainline contracts.  The only algorithmic
piece retained here is the still-distinct `parallax_stratified` camera-pair
proposal rule.

## Ported scope

The optional rule keeps a per-camera nearest-view reserve, fills the remaining
local slots at log-spaced expected-parallax targets, and then restores the
exact nearest global pair budget.  It consumes only mapping poses and one
positive finite median mapping depth per camera.  Descriptor rows, pair
matches, Track identities, test images, and deployed Map rows are unavailable
to the selector.

The port is deliberately narrow:

- `nearest` and `parallax_diverse` remain byte-for-byte behaviorally unchanged;
- `parallax_stratified` is opt-in in the current Track factor CLI;
- the current cache, matcher, Track, triangulation, sidecar, and factor lineage
  code is reused instead of restoring the archived runner;
- the requested pair budget must equal the frozen nearest budget;
- no existing P7/P8 mechanism gate accepts the new policy name, so this port
  cannot authorize fullchain, pose, test, or a default change.

## Existing evidence and the two-sided sentinel

The archived Stairs K1024 artifact was reloaded read-only under the current
unchanged broad-Track definition.  It covers all 2,000 mapping queries, has
23,826 broad Tracks, and gives P8's lost query 1933
(`seq-06/frame-000433.color.png`) 149 broad supporting Tracks.  This is direct
evidence that the proposal family can avoid the specific P8 V2 support hole.

That evidence is not a deployment result.  The archived full mapping pose
replay also produced eleven coherent 27--31 cm false poses in
`seq-05/frame-000226--000242`, even while repairing the earlier failure near
frame 251.  The required interpretation is therefore two-sided:

- query 1933 is the missing-support sentinel;
- the seq-05 frame-226--242 window is the coherent-false-pose sentinel.

A future bounded run is useful only if it uses the current frozen input
contracts and reports both sentinels.  Improving the first while regressing the
second is not a repair.

## Gate interpretation

The formal P8 V2 result remains `STOP_SCENE_MECHANISM`; its preregistered gate
is not edited after observing the result.  For exploration, a softer mechanism
scorecard may report exact lost/added query IDs and their baseline support, but
it is non-authorizing.  Lineage, mapping-only scope, pair budget, and producer
identity remain exact.  Deployment still requires a new preregistered
cross-scene Track and pose/tail gate.

The current conclusion is therefore **GO to one optional proposal-only factor
replay, STOP on default integration**.  The port itself is not evidence that
P8 is fixed and does not reopen P8 fullchain or test evaluation.
