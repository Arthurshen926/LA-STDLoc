# V4 exact frontend acceleration bounded probe

## Decision

Keep the existing float32 SuperPoint frontend unchanged.  None of the bounded
fixed-resolution engineering candidates provided a reproducible material gain
while preserving the frozen sparse-output contract.

The probe used only real **mapping** images from Cambridge ShopFacade and
GreatCourt.  Both datasets use one fixed 1920 x 1080 resolution.  Detector,
keypoint budget, descriptor precision, global Top-1 matching, and PoseLib were
held fixed.  No test image was read.

## Results

| candidate | observed latency result | equivalence | decision |
|---|---:|---|---|
| channels-last + cuDNN benchmark | ShopFacade 44.09 -> 42.20 ms | keypoint rows, scores and descriptors changed; GreatCourt returned 2015 vs 2014 rows | Stop |
| contiguous cuDNN benchmark | ShopFacade 46.15 -> 44.57 ms (1.035x) | exact in the bounded image | Stop: too small for a process-global policy |
| `torch.compile` dense forward | ShopFacade 46.70 -> 38.41 ms (1.216x) | sparse rows/scores/descriptors changed | Stop |
| CUDA Graph dense forward | ShopFacade 25.25 -> 28.62 ms (0.882x) | exact | Stop: slower |
| reusable input buffers | interleaved ShopFacade 53.34 -> 53.54 ms; GreatCourt 67.11 -> 68.16 ms | rows, scores, descriptors, Top-1, pose and inliers exact | Stop: no reproducible gain |
| reusable sparse output buffers | not run | dynamic row count makes returned-storage ownership unsafe | Stop |

The reusable-input prototype initially appeared faster when all control calls
ran before all candidate calls.  A 20-pair AB/BA interleaved replay removed the
ordering and GPU-clock bias and showed no gain in either scene, so the prototype
was removed rather than promoted on a misleading microbenchmark.

CUDA Graph capture was limited to the static dense SuperPoint forward.  Sparse
NMS, thresholding, Top-K selection, and descriptor sampling remained outside
the graph, so dynamic keypoint output ownership was never weakened.  This safe
form was exact but slower.

Machine-readable measurements and the exact/Stop reasons are recorded in
`docs/evidence/v4_frontend_exact_acceleration_probe.json`.

## Consequence

The remaining online bottleneck is not an obvious fixed-resolution SuperPoint
layout or allocation bug.  The already merged exact matcher optimization
remains valid, while hard-scene latency is dominated by PoseLib and therefore
depends primarily on improving correspondence quality or on real frame-level
GPU/CPU pipelining.  FP16, detector replacement, and keypoint reduction remain
outside this engineering probe.
