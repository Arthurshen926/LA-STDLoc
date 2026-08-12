# Offline XFeat Arm-B producer

## Scope

`scripts/materialize_xfeat_arm_b.py` materializes only the descriptor-identity
ceiling probe (Arm B). It runs XFeat on mapping images, samples the locked 64D
dense field at every exact cached SuperPoint row, and writes a
`lafgs_frontend_ceiling_probe_cache` version-1 payload for
`scripts/audit_frontend_upper_bound.py evaluate --arm descriptor`.

It does not run a candidate detector, LighterGlue, pair matching, test images,
map rebuilding, pose evaluation, or any network operation. The implementation
is CPU-only; `--device` accepts only `cpu`.

## Fail-closed inputs

The command requires the prepared dataset root, version-11+ query-cache
signature, complete-positive teacher, clean external XFeat Git worktree, exact
`xfeat.pt`, and explicit expected Git/SHA identities. Before inference it
requires:

- the cache signature to equal the canonical SHA256 of its signature payload;
- the serialized cache and teacher files to equal their explicit command-line
  SHA256 values, and later requires the consumer to resolve those same paths
  and hashes from the probe;
- the cache and ordered teacher names to equal the dataset's complete mapping
  split, with no test name or partial teacher;
- one valid `native_keypoints`/descriptor/mask/HW record per mapping image;
- identical fixed detector K, NMS radius, pixel-center, valid-mask, and native
  coordinate contracts in every record;
- every reference row to be finite, in bounds, valid-mask legal, and bound by
  its exact tensor SHA256;
- the dataset path, image directory, resolution, and longest-edge transform to
  equal the query-cache lineage;
- the recreated object/sky/distortion mask to exactly equal the cached native
  mask;
- a completely clean external Git worktree at the expected parent commit and
  XFeat tree; and
- exact SHA256 for `model.py`, `interpolator.py`, `xfeat.py`, and `xfeat.pt`;
- tracked Git blob IDs for those files and the XFeat `LICENSE`, all under the
  exact locked XFeat tree; and
- a finite, strict CPU state-dict load.

Only `encoders/XFeat/weights/xfeat.pt` is accepted. The adjacent
`xfeat-lighterglue.pt` cannot be supplied even if its hash is known. Verified
external source is compiled in memory, so the producer does not create
`__pycache__` or otherwise dirty the external worktree. The artifact and every
source image are checked again against concurrent changes before output.

## Image and coordinate contract

For each mapping query, the producer reproduces the frozen input in two locked
stages:

1. The prepared image is loaded as float `[0,1]`, the query-cache `resolution`
   transform is replayed with bilinear `align_corners=false`, and the raw
   object/sky/distortion mask is resized with nearest sampling.
2. The query-cache `longest_edge` transform is replayed, its resulting HW must
   exactly equal `native_input_hw`, and RGB is multiplied by the verified
   native valid mask.

The audited XFeat preprocessing is then applied exactly:

```text
Hx = floor(Hn / 32) * 32       Wx = floor(Wn / 32) * 32
rh = Hn / Hx                  rw = Wn / Wx
XFeat row = (native_x / rw, native_y / rh)
```

The masked native RGB is bilinearly resized to `(Hx, Wx)` with
`align_corners=false`. XFeat's stride-8 64D field is L2-normalized before
sampling. The external locked `InterpolateSparse2d('bicubic')` applies its
official normalization
`(2*x/(Wx-1)-1, 2*y/(Hx-1)-1)` with `grid_sample(...,
align_corners=false)`, and every sampled row is L2-normalized again.

The output stores float32 `[N_reference_rows,64]` descriptors plus query
index/name, exact row indices and hashes, original/native/XFeat HW, both scale
ratios, interpolation semantics, image/mask/preprocessed-RGB hashes,
checkpoint/code hashes, and `mapping_only=true`, `uses_test_queries=false`.
Before writing, the existing consumer validates schema, K, query set, row hash,
shape, finiteness, nonzero rows, independent-frontend family, and weight SHA.

## Locked local XFeat command

The following command records the locally inventoried candidate. Replace only
the dataset/cache/teacher/output paths. This command is a provisioned recipe;
no real-scene run was performed while implementing it.

```bash
PYTHONPATH=/root/STDLoc \
  python -m scripts.materialize_xfeat_arm_b \
  --dataset /locked/prepared/scene \
  --query-cache /locked/scene/query_cache.pt \
  --expected-query-cache-sha256 QUERY_CACHE_SHA256 \
  --teacher /locked/scene/complete_positive_teacher.pt \
  --expected-teacher-sha256 TEACHER_SHA256 \
  --xfeat-worktree /mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean \
  --weights /mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean/encoders/XFeat/weights/xfeat.pt \
  --expected-weights-sha256 0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b \
  --expected-parent-commit b28d53258ab4461ba1a02eaa60ef504e9b82b9ab \
  --expected-xfeat-tree 4f804566cb1cf72469b7d7174fba9308885c5c5a \
  --expected-model-sha256 d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964 \
  --expected-interpolator-sha256 d63a6163eb6fff81e8720231f62537a42a69fccb44dc8851b04de5115daab4da \
  --expected-wrapper-sha256 f1b0f73c77e34381a46578866bb1531b98180e8d870c0fc61fbfdbd29ac64f31 \
  --device cpu \
  --output /locked/probes/scene_xfeat64_arm_b.pt
```

The mechanism audit remains a separate read-only step:

```bash
PYTHONPATH=/root/STDLoc \
  python -m scripts.audit_frontend_upper_bound evaluate \
  --state /locked/scene/compact_anchor_map.pt \
  --query-cache /locked/scene/query_cache.pt \
  --teacher /locked/scene/complete_positive_teacher.pt \
  --probe-cache /locked/probes/scene_xfeat64_arm_b.pt \
  --arm descriptor \
  --output /locked/reports/scene_xfeat64_arm_b.json
```

An existing output is never replaced unless `--overwrite` is explicit. Source
artifacts and any path inside the external XFeat worktree can never be used as
the output target, including with `--overwrite`.

## Reviewed Stairs command

The admissible Stairs reference is the fresh, exact-equivalent K1024/NMS4
cache. The historical V3 cache is deliberately inadmissible because its
signature predates the version-11 NMS registration. A read-only preflight
confirmed 2,000/2,000 mapping query names, global `(K,NMS)=(1024,4)`, and zero
out-of-bounds V3-teacher rows against the fresh cache. The locked inputs are:

- fresh query cache SHA256
  `6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9`;
- V3 complete-positive teacher SHA256
  `3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81`;
- V3 compact state SHA256
  `5f754ace648336d9f1fca381f29cd7f6164a217ca05b506644f21929e4a9e620`;
- fresh cache signature
  `286b992273c10d1b15bb616e7cc9499b32911204b9dee915683bb10df56b7342`.

After this producer commit is reviewed and merged, materialize Arm B with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/root/STDLoc \
  /root/miniconda3/envs/g4splat/bin/python -m scripts.materialize_xfeat_arm_b \
  --dataset /mnt/pool/sqy/datasets/7Scenes_pgt_lafgs_v1/stairs \
  --query-cache /mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt \
  --expected-query-cache-sha256 6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9 \
  --teacher /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/complete_positive_teacher.pt \
  --expected-teacher-sha256 3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81 \
  --xfeat-worktree /mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean \
  --weights /mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean/encoders/XFeat/weights/xfeat.pt \
  --expected-weights-sha256 0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b \
  --expected-parent-commit b28d53258ab4461ba1a02eaa60ef504e9b82b9ab \
  --expected-xfeat-tree 4f804566cb1cf72469b7d7174fba9308885c5c5a \
  --expected-model-sha256 d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964 \
  --expected-interpolator-sha256 d63a6163eb6fff81e8720231f62537a42a69fccb44dc8851b04de5115daab4da \
  --expected-wrapper-sha256 f1b0f73c77e34381a46578866bb1531b98180e8d870c0fc61fbfdbd29ac64f31 \
  --device cpu \
  --output /mnt/pool/sqy/lafgs_xfeat_arm_b_20260813/stairs/xfeat64_descriptor.pt
```

Record the printed probe SHA256, verify the compact-state SHA above, then run
the mapping-only consumer. The consumer now fails if the query-cache or teacher
serialized path/SHA differs from the producer-bound source, including a
different teacher with the same schema:

```bash
PYTHONPATH=/root/STDLoc \
  /root/miniconda3/envs/g4splat/bin/python -m scripts.audit_frontend_upper_bound evaluate \
  --state /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/anchor_map_step_1520.pt \
  --query-cache /mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt \
  --teacher /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/complete_positive_teacher.pt \
  --probe-cache /mnt/pool/sqy/lafgs_xfeat_arm_b_20260813/stairs/xfeat64_descriptor.pt \
  --arm descriptor \
  --output /mnt/pool/sqy/lafgs_xfeat_arm_b_20260813/stairs/xfeat64_descriptor_report.json
```

## Synthetic verification

The producer tests use a clean temporary Git checkout, strict fake state dict,
prepared mapping/test split, non-divisible `33x65` native image, official
`32x64` XFeat resize, and real consumer validation. They cover exact bicubic
sampling geometry, `[N,64]` shape/norm, mapping-only exclusion, row-registry
hash corruption, query signature/test-set rejection, checkpoint/code SHA
mismatch, dirty worktree rejection, and pair-checkpoint rejection. Tests use
CPU tensors and synthetic images only.
