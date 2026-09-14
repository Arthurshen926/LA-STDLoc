# V7 immutable method contract

V7 is **Safeguarded Episodic Closed-Loop Map Distillation**. Its deployed
localizer is permanently frozen to:

```text
frozen native SuperPoint -> exact global cosine Top-1 -> one standard PoseLib
```

The formal method obeys all twelve invariants below. The machine-readable copy
lives in `common/v7_contracts.py`; the formal runner checks it against
`configs/v7_safe_closed_loop.yaml` before reading or writing an artifact.

1. Source mapping RGB is never used.
2. The detector is never trained.
3. No independent learned matching scorer is trained.
4. No query adapter, context network, or stronger online feature is used.
5. Multi-prototype maps are forbidden.
6. Each Anchor has one stable ID, one xyz, and one descriptor.
7. Gaussian centers and rendered depth are never PnP coordinates.
8. Feedback queries never enter Tracks, observation CSR, or descriptor banks.
9. Feedback descriptors are never copied into the map.
10. Initialization and every update call the same Selector.
11. Online localization is always the frozen plant stated above.
12. Formal test queries cannot update the map, tune thresholds, or select candidates.

## Phase gate

Every phase consumes hash-bound artifacts from the preceding phase. P0 must
reproduce the frozen baseline compact map byte-for-byte and tensor-for-tensor,
preserve the online deployment contract, and pass the recursive formal import
audit before later phases are enabled. Operating-point selection and descriptor
control consume only non-test render batches. The real test split remains sealed
until the operating point and any P6 rollback decision have been frozen.

The V6 proposal, prototype, LOO, sensor-variant, acceptance, and legacy
closed-loop modules remain available only for historical reproduction. The V7
formal runner cannot import them.

## Render-quality certificate v2

SuperPoint always receives the complete, unmasked float16-replayed RGB. Render
quality is fused only after detection at the 2,048 keypoint rows. A row is
eligible for feedback only when alpha/depth support is valid, it is outside the
image border and depth-discontinuity bands, it lies near reproducible local RGB
structure, and it is not an extreme 2DGS-distortion tail sample.

RGB structure is a resolution-scaled local gradient/variance support prior; it
is not an artifact classifier and is never a global reliability weight. The
2DGS distortion veto is deliberately restricted to values above both the
robust `median + 20 MAD` threshold and the per-render 99.5th percentile. This
prevents a broad secondary distortion mode from being mislabeled as an extreme
outlier. Every row persists separate reason bits for pixel support, depth
discontinuity, border, extreme distortion, and low RGB structure.

## Precision-deficit feedback

P4 retains the nominal-success, representation-deficit, coverage-deficit, and
unreliable-query routes and adds `precision_deficit` for a successful pose whose
correspondence set is measurably inferior to correspondences already available
in the deployed map. The diagnostic is strictly offline: it uses GT pose and
render depth only after the frozen RGB plant has returned its pose.

An alternative set must contain at least 16 Anchor-unique correspondences,
occupy at least six cells of a 4x4 image grid, and differ from the deployed
Top-1 winners on at least eight query rows. It is replayed through the same
single standard PoseLib wrapper and seed. A deficit requires both translation
improvement of at least `max(0.05cm, 10%)` and rotation improvement of at least
`0.005deg`. P5 still requires consistent evidence from at least two independent
pose families for each changed Anchor. Confirmation queries may expose this
diagnostic but are always update-ineligible.

Feedback descriptors only assign weights to original mapping observations.
The controller reconstructs the one descriptor of an already deployed Anchor,
caps its angular change at five degrees, and cannot change map cardinality or
copy a feedback descriptor into the map. Every changed proposal must pass a
disjoint P6 confirmation batch; failure performs an exact SHA-bound rollback.

## Full-map continuous-feedback diagnostic gate

The follow-up mainline starts from the complete 200,255-Anchor projective map;
it disables initialization-map selection and the Full-to-Large teacher/student
path. Before any new planner or controller is enabled, a read-only P0 diagnostic
may render the Gaussian model at test-camera poses and intrinsics. This is an
explicitly non-formal, transductive pose-distribution oracle: it may read test
camera metadata, but it must not open test RGB, select or mutate a map, tune a
threshold, or authorize an update.

P0 is a hard causal gate. If certified render-at-test-pose localization is also
poor, the conditional aggressive planner and continuous observer may be
enabled. If it is near-perfect while localization of real RGB at exactly the
same poses is poor, rendered-query feedback is stopped for the objective of
improving real-test localization. Conditional stages are not run merely to
complete a checklist. Their thresholds and actions are frozen in
`configs/v7_full_continuous_mainline.yaml`.
