# Long-LRM quick-prior experiment

The ShopFacade experiment uses 32 mapping images selected by camera-position
and viewing-direction farthest-point sampling, then sorted in trajectory order.
Camera intrinsics and poses come from the existing Cambridge COLMAP model.
The 231 mapping images are available to the subsequent 1000-step GraphDeco
refinement. Test images are used only for reporting rendering/localization.

## Frozen inputs

- Repository: `/root/third_party_lafgs_priors/Long-LRM`, commit
  `0e89431a964b8a7e2fdec24e3ca5ec74a26dcafc`.
- Checkpoint: `arthurhero/llrm_checkpoints/dl3dv_i540_32input_8target_aug/checkpoint_000010000.pt`;
  SHA256 `51c35ad12a900fe46925a63c64a370345456debde77a8b900d319bcaf745e3a2`.
- Runtime: `/root/miniconda3/envs/long_lrm22`; PyTorch 2.2.0+cu121,
  xformers 0.0.24, gsplat 1.4.0+pt22cu121, causal-conv1d 1.4.0,
  mamba-ssm 2.2.2, transformers 4.44.2. This differs from the upstream
  PyTorch 2.1.2 recipe and must be recorded when reporting reproduction.
- Input manifest:
  `/mnt/pool/sqy/anygsloc_fast_prior_20260904/ShopFacade/long_lrm_input_pose32/anygsloc_long_lrm_input.json`.
- Initial normalized PLY SHA256:
  `2d8f8f858df3686b9e6bc5da89ee3374aecb868f16b34339a47c94e95b286347`.
- World PLY SHA256:
  `e9ec27ab4e0e9bd9e4c81d4d1a6c1e13056aa430c2f20f2457334628efefffcb`.

## Initial artifact audit

The completed September 4 artifact contains 2,575,078 Gaussians after the
upstream pruning and opacity threshold. World export preserves all rows and
transforms centers, scales, covariance rotations, and degree-3 spherical
harmonics. The adapter normalization differs from the upstream Torch
normalization by at most 1.79e-7 on the actual 32 cameras.
An actual first-camera rendering comparison at 320x180 also passes: normalized
versus exported world-frame RGB mean absolute difference 1.185e-6, P99 3.815e-6
(maximum 0.002858 at isolated rasterization boundaries).

The initial run took 861.73 seconds including preprocessing, loading, forward,
and export. It did not save synchronized forward timing or peak-memory
statistics. Earlier progress-message memory snapshots are not peak measurements.
The upstream pruning includes a random subset; the initial run did not set a
seed. Its frozen PLY is reproducible as an evaluation input, but regeneration
is not guaranteed bit-identical. The runner now seeds subsequent runs and
records memory statistics, synchronized full-forward timing, and upstream
forward timing (explicitly unsynchronized). CPU preprocessing uses eight threads
by default to bound thread oversubscription.

The vectorized writer replaces the upstream tuple assembly. The initial writer
used a GPU float32 opacity predicate; the current writer uses the upstream CPU
float32 predicate to avoid a possible threshold-boundary rounding difference.
The synthetic layout test checks every field against the upstream layout.
These later runner improvements do not mutate the initial experimental PLY.

On 12 uniformly sampled mapping views, raw median PSNR is 14.4543 dB and
SSIM is 0.4451. Visual inspection shows broadly aligned storefronts with blur,
ghosting, and one notably poor close view. This is appearance evidence only.
Report and reference/render/error montage:
`/mnt/pool/sqy/anygsloc_fast_prior_20260904/ShopFacade/view_audit_long_lrm_raw_mapping12/`.

## Refinement and evaluation

Run the same 1000-step optimizer settings documented in
[the i3DGS recipe](fast_i3dgs_prior.md), using the Long-LRM world PLY as
`--init_ply`. The initial GPU-resident image run ran out of memory near step 50.
The retry uses `--data_device cpu` and
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`; resolution, Gaussian count,
losses, and learning rates are unchanged. The retry directory is
`long_lrm_globaljoint_d1_ft1000_cpuimages` under the experiment root.

Both raw and refined priors use the standard high-capacity AnyGSLoc Base
pipeline, without offline self-localization feedback. Raw rendering uses the
Long-LRM white background; GraphDeco refinement uses the same black background
as the i3DGS experiment. Always report that distinction with appearance metrics.
The raw evaluation directory is `anygsloc_base_long_lrm_raw_pose32`.

Raw test localization completed on 103 images: median TE 8.4492 cm, median RE
0.24684 degrees, R5 15.534%, zero >1 m failures. Mean runtime was 407.99 ms,
including 345.92 ms RANSAC; mean solver inlier ratio was 11.505% and mean
hypothesis count 32,031. The contemporary i3DGS+1k reference replay exactly
reproduced its previous pose metrics (6.9438 cm / 0.16495 degrees / 24.272% R5),
with 100.55 ms mean runtime and 3,030 mean hypotheses. Runtime measurements
include cold-start effects and were collected while training on a different GPU;
they are descriptive pipeline timings, not isolated deployment benchmarks.
The reference replay is `i3dgs_reference_replay_20260907` under the experiment root.

Raw test-view audit (same 12-view sampling rule): median PSNR 13.8707 dB,
SSIM 0.41722. No reconstruction parameters were selected using this audit.

The 1000-step CPU-image refinement completed successfully in 458 seconds of
training-loop time. Refined test-view median PSNR is 14.1954 dB and SSIM 0.45983,
compared with 13.5121 dB / 0.43636 for i3DGS+1k on exactly the same 12 images.
Both refined audits use black background. This rendering improvement alone
does not establish better localization.

## Entry points

Use `python -m scripts.prepare_long_lrm_cambridge --dataset <Cambridge scene>
--images processed --output <fresh input directory>` to prepare input.
Run `python -m scripts.run_long_lrm_cambridge --long-lrm-root <checkout>
--prepared <input directory> --checkpoint <checkpoint> --output <fresh output>
--seed 2026 --cpu-threads 8` in the `long_lrm22` environment.
Use `python -m scripts.export_long_lrm_prior --source-ply <raw gaussians.ply>
--prepared-manifest <input directory>/anygsloc_long_lrm_input.json
--output-ply <world PLY> --output-manifest <export report>` before any world-frame
rendering or Anchor construction. Raw PLY uint8 normal fields are accepted by
this adapter; the exported world PLY uses the standard all-float field layout.

Evaluate with `scripts.run_anygsloc_scene` and explicit `--gaussian-type 3dgs
--sh-degree 3`; add `--white-background` for raw, omit it for the black-background
refined prior. The mainline config keeps both offline feedback and optional
online refinement disabled for this prior comparison.

The comparison is a pipeline comparison: Long-LRM consumes 32 images for
initialization while i3DGS consumes its own mapping stream. Equal refinement
steps do not imply equal initialization data or compute. Historical AnySplat
results from other scenes cannot establish a same-scene ranking.

## Completed localization comparison (September 7)

All rows use the same 103 test queries, PoseLib seed 2026, and feedback-free
AnyGSLoc Base settings. i3DGS was replayed with the current evaluator and its
poses exactly match the historical output. The refined Long-LRM run was also
replayed on an otherwise idle GPU 2 and all 103 poses match its first evaluation.

| Prior | Anchors | Median TE (cm) | Median RE (deg) | R5 | Mean localization (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long-LRM raw, 32 views | 39,426 | 8.449 | 0.24684 | 15.534% | 407.99 |
| Long-LRM + 1000 steps | 31,894 | 7.075 | 0.19617 | 23.301% | 95.74 |
| i3DGS + 1000 steps | 24,024 | 6.944 | 0.16495 | 24.272% | 100.55 |

No row has a >1 m failure. Refinement improves translation on 67/103 queries;
paired R5 changes are 12 gains and four losses. Its solver inlier ratio rises
from 11.505% to 24.257%, and mean RANSAC hypotheses fall from 32,031 to 3,434.
The lower solve cost explains most of the localization-time improvement.
Refined P90 localization time is 118.24 ms, so the mean alone does not establish
a hard 10 Hz deadline. Cross-method differences of a few milliseconds should
not be treated as significant under the shared-host timing conditions.

Long-LRM+1k is usable and approaches i3DGS+1k, but does not beat its localization
accuracy in this experiment. Better PSNR/SSIM did not imply better pose accuracy.
Keep i3DGS+1k as the present fast-prior reference and retain Long-LRM+1k as a
working alternative; broader model rankings need more scenes. The initial
14.4-minute reconstruction and 7.6-minute refinement also do not support a
seconds-scale end-to-end prior-construction claim for this measured pipeline.

Machine-readable metrics, source paths/hashes, and paired/replay checks are in
[the comparison evidence](evidence/long_lrm_shopfacade_20260907.json).
