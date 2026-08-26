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

P0 is the only enabled phase. It must reproduce the frozen baseline compact map
byte-for-byte and tensor-for-tensor, reproduce every non-timing field of every
StMarysChurch test query, preserve the online deployment contract, and pass the
recursive formal import audit. Any P0 failure blocks P1 and all later method work.

The V6 proposal, prototype, LOO, sensor-variant, acceptance, and legacy
closed-loop modules remain available only for historical reproduction. The V7
formal runner cannot import them.
