# P7 Stairs mapping-density single-factor gate

## Decision

**No-Go for fixed `K_mapping=2048`; stop before function-graph, compact Map,
metric refresh, and pose.** Doubling native mapping keypoints makes the Track
funnel much wider, but does not create any additional frozen high-confidence
Track (`70 -> 70`) and worsens median covariance in the broad usable set by
22.86%. The default indoor policy therefore remains V3-equivalent `K=1024`;
`K=2048` remains an explicit factor override, not a paper-mainline default.

This is a mapping-only experiment. It uses all 2,000 Stairs mapping images and
no test query.

## Causal contract

Both arms replay the same frozen zero-step Stairs Track invocation with:

- native SuperPoint frontend and `NMS=4`;
- frozen nearest-6 camera pairs (7,450 candidate/matched pairs);
- the same pair budget, match/epipolar/triangulation thresholds, seed 2026,
  initial state, visibility cache, RGB prior, selector, and descriptor policy;
- repository commit `7415f94edb3ee57d64571140e92e46c444131009` and identical
  experimental diff SHA-256
  `e222482d001622efe8c78852a87a5e57e1bbce0d61e2a29f661917f56a39cf80`.

Only query-cache/output paths and `native_keypoint_count`, `max_observations`,
and `validation_observations` change from 1024 to 2048. The complete checked-in
cache audit is
[`p7_stairs_density_cache_pair_contract.json`](evidence/p7_stairs_density_cache_pair_contract.json):
the query registry is identical; signature payloads are identical except for
K; all 2,000 rows attest their requested K and NMS=4; pose, intrinsics, depth,
and alpha are exact; and every K=1024 sparse row is an exact prefix of K=2048.

The control cache was freshly re-extracted at K=1024/NMS=4 from the frozen V3
cache. Its exact refresh audit is
[`p7_stairs_density_sparse_refresh_equivalence.json`](evidence/p7_stairs_density_sparse_refresh_equivalence.json):
all Track inputs and alpha fields match the legacy cache in 2,000/2,000
queries. Independently, its full Track funnel exactly reproduces frozen V3,
including all edge, Track, observation, and triangulation counts.

## Track funnel

The complete machine-readable result is
[`p7_stairs_density_track_funnel_factor.json`](evidence/p7_stairs_density_track_funnel_factor.json).

| Mapping-only quantity | K=1024 control | K=2048 high | High/control |
|---|---:|---:|---:|
| Native keypoints requested | 2,048,000 | 4,096,000 | 2.000x |
| Raw reciprocal epipolar edges | 1,760,615 | 3,222,251 | 1.830x |
| Accepted cycle edges | 76,735 | 106,461 | 1.388x |
| Accepted chain edges | 1,500,736 | 2,716,470 | 1.810x |
| Accepted edges total | 1,577,471 | 2,822,931 | 1.790x |
| Conflict-rejected edges | 183,144 | 399,320 | 2.180x |
| Tracks | 105,782 | 227,579 | 2.151x |
| Track observations | 1,201,669 | 2,336,552 | 1.944x |
| Triangulated Tracks | 17,798 | 36,732 | 2.064x |
| Broad usable Tracks | 5,079 | 7,833 | 1.542x |
| Strict usable Tracks | 2,023 | 2,575 | 1.273x |
| Frozen high-confidence Tracks | **70** | **70** | **1.000x** |
| Broad covariance median (m²) | 3.8308e-4 | 4.7066e-4 | **1.229x** |
| Broad covariance p90 (m²) | 1.3512e-3 | 1.3979e-3 | 1.035x |
| Broad parallax median | 2.9999° | 2.9704° | 0.990x |

The preregistered mechanism gate passes six of seven checks. It fails the
required `broad covariance median <= 1.10x` check at 1.2286x. This failure is
not a marginal count fluctuation: high K roughly doubles raw evidence and
Track count while leaving the strongest 70 identities unchanged. The added
evidence mainly populates lower-conditioned components, so density alone does
not address the indoor identity/geometry bottleneck.

## Exact reproduction

The K=1024 cache was rebuilt by rerunning the native detector on every mapping
image, rather than editing legacy metadata:

```bash
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/cybersim_agent/bin/python \
  -m scripts.refresh_mapping_sparse_cache \
  --dataset /mnt/pool/sqy/datasets/7Scenes_pgt_lafgs_v1/stairs \
  --source-cache /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/query_cache.pt \
  --output /mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt \
  --mapping-keypoints 1024 --nms-radius 4 --device cuda
```

Each Track arm used the same runner and verified extension environment; replace
`<K>`, `<cache>`, and `<output>` with the paired values above:

```bash
PATH=/root/miniconda3/envs/cybersim_agent/bin:$PATH \
LD_LIBRARY_PATH=/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-} \
CUDA_VISIBLE_DEVICES=2 \
/root/miniconda3/envs/cybersim_agent/bin/python \
  -m scripts.run_mapping_density_track_factor \
  --manifest /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/tracks_refined/reproducibility_manifest.json \
  --query-cache <cache> --output-dir <output> \
  --mapping-keypoints <K> --nms-radius 4 \
  --python /root/miniconda3/envs/cybersim_agent/bin/python
```

The valid control wrote its invocation at `20:10:30+08:00` and payload at
`20:20:52+08:00` (10m22s). High wrote them at `20:22:27+08:00` and
`20:35:24+08:00` (12m57s). GPU2 was confirmed released at
`2026-08-12T20:35:39+08:00`.

Payload SHA-256:

- K=1024: `0710953067a517a23ca4a0cb9334c62d663e1091a54a4c73c6d86e32fb279af3`
- K=2048: `5f92599a9e17b3321419a166b10722ca60a355c85810798a99709db3bc1fbcb9`

Reproducibility-manifest SHA-256:

- K=1024: `3838ba24fd7eeebe59bbfae27b58c5fd4fb3099b8f0cee129f829ceac91b1a53`
- K=2048: `40fcc1b66cb3ac3935beb86c92dae01227f5a2838b9ad70f030a8783817c497d`

## Environment-only failed attempts

The first control attempt reached post-Track G3 provenance, then failed before
payload serialization because `ninja` was absent from the subprocess PATH
(about 17 minutes, nonzero exit, no scientific payload). A second attempt with
PATH corrected was interrupted during camera loading after a smoke check found
that the system `libstdc++` lacked `GLIBCXX_3.4.29` (about 2.5 minutes, no
scientific payload or result manifest). The successful arms used both PATH and
`LD_LIBRARY_PATH` above after `command -v ninja` and a minimal gsplat backend
import passed.

The first strict cache audit was also interrupted after 18 minutes with no JSON
because the old PyTorch runtime launched 94 CPU threads for each small tensor
comparison. It was rerun with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`; the comparison code, inputs, and thresholds were
unchanged and the audit passed. This was a read-only audit performance repair,
not a Track-arm change.

## Consequence for the next mainline

Retain the general `K_mapping`/`K_deployment` separation and full cache
attestation. Do not set indoor `K_mapping=2048` by default and do not combine
this stopped density factor with a pair-policy experiment. The next causal arm
may change only pair geometry at a fixed K and pair budget; its goal should be
to replace low-parallax/poorly-conditioned evidence, not merely add more rows.
