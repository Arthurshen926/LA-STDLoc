# V4 offline frontend batching probe

This bounded probe used only physical GPU 1 and eight ShopFacade mapping
cameras. It did not read test queries, run the 24-scene matrix, or alter the
localization method.

The current producer already keeps one SuperPoint instance alive for its full
camera loop. Reload avoidance therefore has only a 0.235 s in-process ceiling.
Batching is not an exact replacement: for 640x360 inputs, batches 2, 4, and 8
all changed selected keypoint rows. Batch 8 was only 1.007x as fast as the
57.26 ms sequential eight-image reference; batches 2 and 4 were slower.
Consequently persistent GPU services and batched SuperPoint are stopped.

The RGB-only Gaussian loader did expose one genuine loading defect. A
1,202,378-primitive ShopFacade prior without `loc_*` fields previously created
an unused `[N,256]` random localization bank. The legacy probe exceeded 145 s
and about 2.7 GiB RSS before it was aborted. The explicit zero-dimensional
RGB-only contract loaded the same prior in 40.25 s under a busy shared host and
retained `[N,1,0]`; the render path never consumes that tensor. This contract
is retained with a focused regression test.

Renderer batching is not promoted. The prior full 231-view cache spent only
13.10 s rendering, while exact end-to-end batching is already blocked by the
SuperPoint result. Even an impossible zero-cost renderer would save less than
the remaining Track/geometry stages.

Direct packed Observation output is also stopped. The existing list adapter
cost only 1.8 ms for all 231 views. Stacking the 490 MB cache took 5.43 s;
packed serialization saved only 40 ms and 0.08% bytes. Changing the cache
schema would add work and force downstream unpacking.

Fresh-CPU two-way triangulation retains measured 2.03x headroom with exact
fields, but production integration needs a real stage boundary after the GPU
Track producer. It is deferred rather than hidden behind an unsafe post-CUDA
fork or mixed into this frontend-only change.

The machine-readable measurements are in
`docs/evidence/v4_offline_frontend_batch_probe_20260820.json`.
