# Original30k, state-exact continuation v3

## Claim and finite route
Stage1 executes original steps1–14999 and saves one complete safe-boundary checkpoint. Stage2 is separately reviewed after the actual stage1 checkpoint and performance exist, restores that checkpoint bitwise, then executes steps15000–30000. No A/A rerun, first8 gate,15101 reference, branching or numerical trajectory threshold. Native floating trajectory equality is explicitly not claimed. A, B, U1 and U2 and all FAIL receipts remain unchanged.

The U1/U2 pair (pair_comparison.json SHA53c0c0c96a87e18a27dc35236ba46fe0ebf977473a926659c3685c7d4aa5ba7a; independent receipt97eef6a42ad60811e3813cea4da069baf4e6df7952e3995d74d32bbc94836a04) showed21 light and22 full-state field differences under identical native settings. Loss maxima were small (L1 1ULP,DSSIM4ULP,total2ULP), but endpoint max absolute position difference0.0372038 source units and opacity0.178795 show loss proximity is no model-equivalence guarantee. No tolerance is inferred or adopted.

## Science and complete state
The two instrumented trainers are byte-identical to the previously reviewed v2 copies. Original trainer SHA5c5c7d924675dc3ad743208b3e2ca35d3d064f30892338845ff2c08a30b3a630 remains unchanged. The30k scheduler horizon, seed0,20 chart/512dense cameras, learning rates, optimizer target, sampling and events are unchanged. Step30000 retains the native skip-optimizer-step/skip-zero-grad behavior and is not represented as a resumable state.

Checkpoint transport functions and schema remain unchanged. The complete state includes Gaussian parameters; ordered optimizer bindings and Adam states; actual mip and flag; all densification buffers; scheduler current and next LR; Python/NumPy/Torch CPU/all-visibleCUDA RNG; both ordered sampling stacks including None versus empty; mutable chart cadence/phase; seven EMAs; static computed caches, camera RGB and masks; pipeline and event history; source/input/science/runtime bindings. At the14999 safe boundary all parameter gradients must be None and this is checked again on capture and restoration; arbitrary nonempty gradients are not serialized.

The restored actual model, optimizer, loop state, next LR, RNG and actual static caches are fully recaptured and typed-byte digested before15000. A durable restore_bitwise_PASS.json is required. No step may execute before this verification. Every step records view, sampling stacks, RNG digest, loss components, LR, phase, point count and event history; skipped, repeated or out-of-order iterations fail.

Finite checks cover every scalar loss, checkpoint floating parameters/Adam/accumulators/loop state, and terminal model/optimizer/buffers/gradients. They inspect without repair, clipping, replacement or numerical threshold. Native training math remains untouched. The audited diagnostic guard is used only for snapshots/copy/hash/equality/serialization/finite inspection, and restores training threads/RNG in finally, including exception paths.

## Input provenance
Training RGB is mapping RGB. User selected native initialization:419837 retained dense SfM points include226409 test-supported and33423 test-only points. This is disclosed native initialization, not strict mapping-only initialization. Actual corrected-input receipt SHA9448e9b30ad535b36aee84273aca271884810467c2e8df1acb2b0985fda1d2af remains required.

## First-stage resource preregistration
Predicted3238.9s = observed startup upper rounded530 + historical prefix2169×1.10=2385.9 + observed full-capture upper rounded179 + publication upper rounded24 +120 extra source-hash/finite-check/tail/I/O reserve. This is141.1s below3380 work budget; total supervisor3500s and official Runner3600s. No estimate is a completion guarantee; sharedCPU/I/O and correctedinput point evolution can differ.

The2169s anchor is native15000 progress in /root/MAtCha/output_cambridge_full_retained_v2/StMarysChurch_n20_long_masked_retrain/logs/train_30000_driver.log, SHA d695c4472d21b730b8af937203d555e080b4cdd0ab43a557dfab16b79fd8104e. It conservatively covers the14999 prefix and is not the other faster historical log. Startup/full-capture bounds derive from observedA/U1/U2; guard CPU benchmark speedup is not booked as guaranteed GPU-transfer or publication savings.

First stage produces one static archive (previous sameinput14.462GB), one checkpoint (initial5.613M points3.997GB; actual14999 may be smaller, not assumed), original native model saves, light logs and manifests. Minimum available hostRAM96GiB and disk64GiB are mandatory. Previous observed peakRSS40.912GB;96GiB accommodates restore+fresh caches+CPU snapshots. Static caches are written once and referenced by checkpoint hash.

Single assigned physicalGPU0 UUID GPU-55a12bad-1a5e-1538-5097-3b422bdb7ef1, visible logical0 only. Actual DriverAPI context UUID is proven before/after. Inherited exclusive GPU lease, exclusive fresh output, frozen fullspec body/review/source/input references, deferred signal handling, critical Popen ownership and protected process-group drain are reused from the audited supervisor.

Stage2 is not authorized by the stage1 spec. Its actual14999 point count, dynamic/static archive sizes, stage1 elapsed and available memory/disk must drive a separate frozen forecast below3380 and independent review. It may only restore the uniquely linked successful official stage1 output. Any hard-budget/finite/source/state/schema/iteration failure stops; no retry or alternate cut is automatic.
