# LA_update4: RGB Teacher, Artifact Repair, and Pseudo-Query Loop

## Implemented

- Added an independent `la_artifacts` package:
  - `ArtifactDetector`: produces continuous per-pixel, per-region, contributor, and per-Gaussian artifact evidence from RGB residual, feature residual, alpha coverage, and optional support channels.
  - Added no-target low-texture evidence from both global RGB variance and local gradient magnitude, so synthetic RGB renders that are nearly uniform, smooth, or undertrained are rejected instead of receiving a false zero artifact score.
  - `ArtifactRepair`: converts artifact evidence into non-destructive render-time contributor/opacity suppression.
  - `RgbTeacherManifest`: records RGB-only teacher backend metadata, checkpoint, commands, metric summaries, and NerfBaselines trajectory render commands.
  - `PseudoQueryManifest` / `PseudoTeacherCache` / `PseudoQuerySampler`: records real train RGB and synthetic RGB query episodes, teacher cache keys, artifact scores, repair decisions, and source-balanced sampling.
- Added render-time opacity multipliers in `gaussian_renderer` for both RGB and feature/loc render paths. This lets artifact repair suppress the same bad ray contributors across RGB and feature renders without pruning the reference map.
- Added pseudo-query training support in `train_locaware.py`:
  - `--pseudo_query_manifest`
  - `--pseudo_teacher_cache`
  - `--pseudo_query_real_weight`
  - `--pseudo_query_synthetic_weight`
  - `--pseudo_query_max_synthetic`
  - `--pseudo_query_sources`
- Updated `EpisodeSampler` to prefer `camera.teacher_cache_key`, so synthetic query caches are keyed by manifest query id rather than by temporary image filename.

## New Scripts

- `scripts/prepare_rgb_teacher_manifest.py`
  - Creates the RGB teacher manifest.
  - Defaults to WildGaussians command generation and validation.
- `scripts/prepare_nerfbaselines_colmap_dataset.py`
  - Creates a symlinked NerfBaselines-compatible COLMAP staging dataset.
  - Writes `train_list.txt` / `test_list.txt` from Cambridge `dataset_train.txt` / `dataset_test.txt`, so RGB teacher training does not use official test images.
- `scripts/build_pseudo_query_manifest.py`
  - Builds `train_rgb` records from all Cambridge train images.
  - Builds adjacent-pose `synthetic_rgb` records.
  - Can render synthetic RGB through the current in-repo renderer for smoke/fallback validation.
- `scripts/build_pseudo_teacher_cache.py`
  - Runs full STDLoc teacher localization over accepted pseudo-query records.
  - Stores sparse/dense pose, inliers, errors, stage diagnostics, source, artifact score, and repair action.
- `scripts/repair_render_artifacts.py`
  - Runs artifact detection on rendered train views.
  - Projects artifact evidence to visible Gaussians.
  - Re-renders with non-destructive opacity suppression and writes before/after diagnostics.
- `scripts/run_la_pseudo_query_pipeline.sh`
  - Orchestrates RGB teacher manifest, pseudo-query manifest, teacher cache, optional student training, and optional sparse-only eval.
  - Environment variables control scenes, GPU, synthetic count, cache source filter, train steps, and sampling mix.
  - Used for ShopFacade and OldHospital smoke runs below.

## Verified Smoke

### RGB Teacher / WildGaussians

Install and backend validation:

- Created isolated conda env `/root/miniconda3/envs/nb`.
- Installed `nerfbaselines 1.2.12`.
- Installed the `wild-gaussians` NerfBaselines method backend with CUDA 11.8.
- Built CUDA extensions:
  - `simple_knn`
  - `diff_gaussian_rasterization`
- Verified the CLI registers `wild-gaussians`.

NerfBaselines staging command:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python scripts/prepare_nerfbaselines_colmap_dataset.py \
  --source_path /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  --output /mnt/pool/sqy/stdloc_la_nerfbaselines_datasets/ShopFacade \
  --images_source . \
  --force
```

Result:

- Staging dataset written to `/mnt/pool/sqy/stdloc_la_nerfbaselines_datasets/ShopFacade`.
- `image_count = 334`
- `train_split_count = 231`
- `test_split_count = 103`
- Symlinked `images/seq1`, `images/seq2`, `images/seq3`, and `sparse`.
- NerfBaselines confirmed it loads `/train_list.txt` for training, so official Cambridge test images are not used for RGB teacher training.

1-step actual WildGaussians train smoke:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
/root/miniconda3/envs/nb/bin/nerfbaselines train \
  --method wild-gaussians \
  --data /mnt/pool/sqy/stdloc_la_nerfbaselines_datasets/ShopFacade \
  --output /mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step \
  --backend conda \
  --logger none \
  --set iterations=1 \
  --save-iters 1 \
  --eval-few-iters 1 \
  --eval-all-iters 999999 \
  --disable-output-artifact \
  -v
```

Result:

- Train split loaded from `train_list.txt`: 231 official train images.
- Eval split loaded from `test_list.txt`: 103 official test images, used only for evaluation.
- SIMPLE_RADIAL cameras were automatically undistorted into pinhole inputs.
- 1 train iteration completed with `train/psnr = 10.8972`.
- Single train render eval: `psnr = 13.2242`.
- Single test render eval: `psnr = 12.0660`.
- Checkpoint saved:
  - `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/checkpoint-1/chkpnt-1.pth`
  - `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/checkpoint-1/point_cloud.ply`
  - `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/checkpoint-1/nb-info.json`

100-step WildGaussians train smoke:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
/root/miniconda3/envs/nb/bin/nerfbaselines train \
  --method wild-gaussians \
  --data /mnt/pool/sqy/stdloc_la_nerfbaselines_datasets/ShopFacade \
  --output /mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_100step \
  --backend conda \
  --logger none \
  --set iterations=100 \
  --save-iters 100 \
  --eval-few-iters 100 \
  --eval-all-iters 999999 \
  --disable-output-artifact \
  -v
```

Result:

- 100 train iterations completed with `train/psnr = 11.8620`.
- Single train render eval: `psnr = 12.3465`.
- Single test render eval: `psnr = 12.8103`.
- Checkpoint saved at `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_100step/checkpoint-100`.
- Runtime observation: full-resolution camera undistortion dominated wall time, about 4m53s for 231 train images plus 2m10s for 103 eval images; 100 training iterations took about 59s.

Ready manifest command:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python scripts/prepare_rgb_teacher_manifest.py \
  --scene ShopFacade \
  --source_path /mnt/pool/sqy/stdloc_la_nerfbaselines_datasets/ShopFacade \
  --output /mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/rgb_teacher_manifest.json \
  --output_root /mnt/pool/sqy/stdloc_la_rgb_teacher_smoke \
  --backend wildgaussians \
  --checkpoint /mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/checkpoint-1 \
  --nerfbaselines_bin /root/miniconda3/envs/nb/bin/nerfbaselines \
  --nerfbaselines_backend conda \
  --train_steps 1 \
  --logger none \
  --save_iters 1 \
  --eval_few_iters 1 \
  --eval_all_iters 999999 \
  --disable_output_artifact
```

Result:

- Manifest written to `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/rgb_teacher_manifest.json`.
- `status = ready`
- `validation_ok = true`
- Train/render commands use `/root/miniconda3/envs/nb/bin/nerfbaselines --backend conda`.
- Render command now uses the actual NerfBaselines trajectory entrypoint: `render-trajectory --trajectory {trajectory_json} --output {output_path}`.

### Pseudo-Query Manifest

Manifest-only command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python scripts/build_pseudo_query_manifest.py \
  -s /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  -m /mnt/pool/sqy/stdloc_la_full_runs/ShopFacade_baseline \
  -r 1 -f sp -g 3dgs --images processed --data_device cpu \
  --iteration 30000 \
  --scene_name ShopFacade \
  --output /tmp/la_pseudo_shopfacade_manifest_only.jsonl \
  --synthetic_count 2 \
  --synthetic_image_root /tmp/la_pseudo_shopfacade_synth \
  --render_synthetic_backend none
```

Result:

- `train_rgb:accepted = 231`
- `synthetic_rgb:rejected = 2` because backend was `none`.

Rendered synthetic smoke command:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python scripts/build_pseudo_query_manifest.py \
  -s /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  -m /mnt/pool/sqy/stdloc_la_full_runs/ShopFacade_baseline \
  -r 1 -f sp -g 3dgs --images processed --data_device cpu \
  --iteration 30000 \
  --scene_name ShopFacade \
  --output /tmp/la_pseudo_shopfacade_render_smoke.jsonl \
  --synthetic_count 1 \
  --synthetic_image_root /tmp/la_pseudo_shopfacade_render_smoke \
  --render_synthetic_backend inrepo
```

Result:

- `train_rgb:accepted = 231`
- `synthetic_rgb:accepted = 1`
- The accepted synthetic sample had `artifact_score = 4.5125e-06`, `coverage = 0.9701`, and `repair_action = none`.

WildGaussians synthetic render smoke with the 1-step RGB teacher checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python scripts/build_pseudo_query_manifest.py \
  -s /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  -m /mnt/pool/sqy/stdloc_la_full_runs/ShopFacade_baseline \
  -r 1 -f sp -g 3dgs --images processed --data_device cpu \
  --iteration 30000 \
  --scene_name ShopFacade \
  --output /tmp/la_pseudo_shopfacade_wg_render_smoke_v2.jsonl \
  --synthetic_count 1 \
  --synthetic_image_root /tmp/la_pseudo_shopfacade_wg_render_smoke_v2 \
  --render_synthetic_backend wildgaussians \
  --rgb_teacher_checkpoint /mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/checkpoint-1 \
  --nerfbaselines_bin /root/miniconda3/envs/nb/bin/nerfbaselines \
  --nerfbaselines_backend conda \
  --wildgaussians_render_root /tmp/la_pseudo_shopfacade_wg_render_smoke_v2/_wildgaussians_render \
  --wildgaussians_output_names color \
  --synthetic_accept_score 0.65
```

Result:

- `train_rgb:accepted = 231`
- `synthetic_rgb:rejected = 1`
- The 1-step WildGaussians render was near-uniform gray and was rejected by no-target low-texture evidence:
  - `artifact_score = 0.9894`
  - `reason = artifact_score_rejected`
  - `repair_action = wildgaussians_render`
- This closes a real leakage point: undertrained or broken RGB-teacher renders no longer enter teacher cache/student training just because no target residual is available.

Before adding low-texture evidence, the same 1-step render was incorrectly accepted with `artifact_score = 0.0`; running teacher cache on that bad synthetic sample produced `stage_counts = {"sparse_failure": 1}`.

WildGaussians synthetic render smoke with the 100-step RGB teacher checkpoint:

- Global-std-only evidence initially let this sample pass with `artifact_score = 0.3162`.
- Teacher cache on that accepted sample failed at sparse localization:
  - `stage_counts = {"sparse_failure": 1}`
  - log reason: `[SKIP] No enough matches`
- After adding local-gradient evidence, the same 100-step render is rejected:
  - `artifact_score = 0.9468`
  - `reason = artifact_score_rejected`
  - manifest summary: `{'synthetic_rgb:rejected': 1, 'train_rgb:accepted': 231}`
- Image statistics explain the failure mode:
  - 1-step WildGaussians render: global RGB std about `0.0018`, local gradient mean about `1.05e-05`
  - 100-step WildGaussians render: global RGB std about `0.0206`, local gradient mean about `1.06e-04`
  - in-repo STDLoc synthetic render: global RGB std about `0.2418`, local gradient mean about `0.0145`

### Teacher Cache

Command:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python scripts/build_pseudo_teacher_cache.py \
  -s /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  -m /mnt/pool/sqy/stdloc_la_full_runs/ShopFacade_baseline \
  -r 1 -f sp -g 3dgs --images processed --data_device cpu \
  --iteration 30000 \
  --cfg configs/stdloc_cambridge.yaml \
  --manifest /tmp/la_pseudo_shopfacade_render_smoke.jsonl \
  --output /tmp/la_pseudo_shopfacade_teacher_cache_smoke.pt \
  --summary_json /tmp/la_pseudo_shopfacade_teacher_cache_smoke.json \
  --max_queries 1
```

Result:

- `count = 1`
- `stage_counts = {"teacher_ok": 1}`

## Tests

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_episode_sampler \
  tests.test_la_artifacts \
  tests.test_render_artifact_weights \
  tests.test_teacher_stage_diagnostics \
  tests.test_full_script_args
```

Result:

- `Ran 66 tests`
- `OK`

Also passed:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  la_artifacts/*.py \
  scripts/prepare_rgb_teacher_manifest.py \
  scripts/prepare_nerfbaselines_colmap_dataset.py \
  scripts/build_pseudo_query_manifest.py \
  scripts/build_pseudo_teacher_cache.py \
  scripts/repair_render_artifacts.py \
  train_locaware.py \
  localization_training/episode_sampler.py \
  gaussian_renderer/__init__.py
```

and `git diff --check`.

## Current Limits

- WildGaussians/NerfBaselines is installed and 1-step/100-step ShopFacade train smoke runs succeed, but a high-fidelity RGB-only teacher map has not been trained yet. The 100-step model is still too smooth for STDLoc sparse matching.
- Synthetic RGB generation now supports both the in-repo renderer and WildGaussians/NerfBaselines `render-trajectory`.
- Artifact repair is still render-time contributor suppression for in-repo renders. NerfBaselines CLI renders can be detected/rejected, but contributor-level repair requires a lower-level WildGaussians render API or exported contributor metadata.
- Full-resolution NerfBaselines dataset loading performs COLMAP undistortion on first use; on ShopFacade this took several minutes. Long multi-scene jobs should use persistent staged datasets and avoid repeatedly deleting converted outputs.
- Full 100/500-step student ablations are not run in this update. The verified loop is implementation-level plus ShopFacade smoke:
  - all train RGB manifest
  - one in-repo synthetic RGB render/audit/accept
  - one WildGaussians synthetic RGB render/audit/reject for a 1-step undertrained RGB teacher checkpoint
  - one WildGaussians synthetic RGB render/audit/reject for a 100-step smooth but sparse-unmatchable RGB teacher checkpoint
  - one teacher-cache localization record
  - train-time loader and sampler integration
- GPU2 still has a stale 18 GB allocation from a non-existent PID. NVIDIA driver reset failed locally, so it should be avoided until the host is rebooted or the driver state is cleared externally.

## Pipeline Follow-Up

### Script

Added `scripts/run_la_pseudo_query_pipeline.sh`.

Example smoke command:

```bash
SCENES=ShopFacade \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_pseudo_query_smoke \
GPU=0 \
SYNTHETIC_COUNT=1 \
RUN_TEACHER_CACHE=1 \
TEACHER_CACHE_SOURCES=synthetic_rgb \
TEACHER_CACHE_MAX=1 \
RUN_TRAIN=1 \
TRAIN_STEPS=1 \
PSEUDO_QUERY_SOURCES=synthetic_rgb \
PSEUDO_QUERY_REAL_WEIGHT=0.0 \
PSEUDO_QUERY_SYNTHETIC_WEIGHT=1.0 \
RUN_EVAL=0 \
FORCE_TRAIN_COPY=1 \
scripts/run_la_pseudo_query_pipeline.sh
```

### ShopFacade Smoke

Output root:

- `/mnt/pool/sqy/stdloc_la_pseudo_query_smoke/ShopFacade`

Results:

- RGB teacher manifest written.
- WildGaussians is now installed through NerfBaselines and the ready manifest at `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/rgb_teacher_manifest.json` validates successfully against checkpoint `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg_1step/checkpoint-1`.
- Pseudo-query manifest:
  - `train_rgb:accepted = 231`
  - `synthetic_rgb:accepted = 1`
  - synthetic record includes repair metadata:
    - `artifact_score_before_repair = 4.5125e-06`
    - `artifact_score_after_repair = 4.5125e-06`
    - `repair_suppressed_gaussians = 0`
    - `repair_action = none`
- Synthetic-only teacher cache:
  - `count = 1`
  - `stage_counts = {"teacher_ok": 1}`
- 1-step student training with `PSEUDO_QUERY_SOURCES=synthetic_rgb` completed:
  - saved `/mnt/pool/sqy/stdloc_la_pseudo_query_smoke/ShopFacade/student_1step_seed0/point_cloud/iteration_30001/point_cloud.ply`
  - saved `/mnt/pool/sqy/stdloc_la_pseudo_query_smoke/ShopFacade/student_1step_seed0/point_cloud/iteration_30001/loc_state.pt`
  - log line: `[ITER 30001] base 0.150994 loc 0.543287 psnr 19.311`

This verifies the end-to-end path:

`synthetic RGB render -> artifact detect/repair metadata -> teacher cache -> RGB FeatureExtractor query -> teacher_cache_key sparse init -> student loc-aware loss -> checkpoint save`.

### OldHospital Smoke

Command used the same pipeline with:

```bash
SCENES=OldHospital \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_pseudo_query_smoke \
GPU=1 \
SYNTHETIC_COUNT=1 \
RUN_TEACHER_CACHE=1 \
TEACHER_CACHE_SOURCES=synthetic_rgb \
TEACHER_CACHE_MAX=1 \
RUN_TRAIN=0 \
RUN_EVAL=0 \
scripts/run_la_pseudo_query_pipeline.sh
```

Results:

- RGB teacher manifest written.
- Pseudo-query manifest:
  - `train_rgb:accepted = 895`
  - `synthetic_rgb:accepted = 1`
- Synthetic-only teacher cache:
  - `count = 1`
  - `stage_counts = {"teacher_ok": 1}`

### GPU State

- GPU0 and GPU1 are usable and idle after smoke runs.
- GPU2 still shows a stale 18 GB allocation from a non-existent PID. `nvidia-smi --gpu-reset -i 2` failed with an NVIDIA driver reset error and suggested a host reboot. Current experiments should avoid GPU2.

## Update 2026-06-27: RGB Teacher Downscale and Teacher-Pose Gate

### Implementation Delta

- `scripts/prepare_nerfbaselines_colmap_dataset.py`
  - Added materialized image resize staging:
    - `--image_downscale_factor`
    - `--max_image_width`
  - The output manifest now records resize mode, original/staged first image size, copied count, and resized count.
- `scripts/run_la_pseudo_query_pipeline.sh`
  - Added pipeline env passthrough:
    - `NERFBASELINES_IMAGE_DOWNSCALE_FACTOR`
    - `NERFBASELINES_MAX_IMAGE_WIDTH`
    - `WILDGAUSSIANS_RENDER_SCALE`
    - `PSEUDO_QUERY_FILTER_TEACHER_CACHE`
    - `PSEUDO_QUERY_TEACHER_MAX_SPARSE_TE`
    - `PSEUDO_QUERY_TEACHER_MAX_DENSE_TE`
    - `PSEUDO_QUERY_TEACHER_ALLOWED_STAGES`
- `la_artifacts/rgb_teacher.py` and `scripts/build_pseudo_query_manifest.py`
  - WildGaussians trajectory generation now supports `image_scale`.
  - Synthetic records now update `width`/`height` from the actual rendered RGB frame and record `nerfbaselines_render_scale`.
- `la_artifacts/pseudo_query.py`, `localization_training/episode_sampler.py`, and `train_locaware.py`
  - Added teacher-cache quality filtering before pseudo-query sampling.
  - Default filter when cache is present: `max_sparse_te = 100cm`, `max_dense_te = 100cm`.
  - `EpisodeSampler` no longer uses sparse init from cache entries classified as `sparse_failure` or `dense_rescues_sparse`.

### ShopFacade 960px WildGaussians 500-Step Smoke

Staged dataset:

- `/mnt/pool/sqy/stdloc_la_nerfbaselines_datasets/ShopFacade_960`
- `image_count = 334`
- `train_split_count = 231`
- `test_split_count = 103`
- `first_size = [1920, 1080] -> [960, 540]`
- `resized_image_count = 334`

Training output:

- `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg960_500step`
- Checkpoint:
  - `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg960_500step/checkpoint-500`
- Observed timing:
  - train split undistortion: about 1m19s, versus about 4m53s in the earlier full-resolution run
  - eval split undistortion: about 35s
  - 500 training steps: about 53s
- Metrics:
  - `train/psnr = 14.0690`
  - single train render eval `psnr = 14.0732`
  - single test render eval `psnr = 14.1797`

This is a real improvement over the earlier full-resolution 100-step smoke (`train/psnr = 11.8620`), but it is still not a final high-fidelity RGB teacher.

### Synthetic RGB Smoke from 500-Step RGB Teacher

Manifest:

- `/tmp/la_pseudo_shopfacade_wg960_500_render_smoke.jsonl`
- `train_rgb:accepted = 231`
- `synthetic_rgb:accepted = 4`

Synthetic records:

- All 4 WildGaussians renders were `960x540`.
- All 4 recorded `nerfbaselines_render_scale = 0.5`.
- All 4 had rendered-RGB-only artifact score `0.0`.

This shows the low-texture detector no longer rejects these as empty/smooth frames. It does not prove teacher usability by itself.

### Teacher Cache on Accepted Synthetic

Cache:

- `/tmp/la_pseudo_shopfacade_wg960_500_teacher_cache.pt`
- `/tmp/la_pseudo_shopfacade_wg960_500_teacher_cache_summary.json`

Result:

- `count = 4`
- `stage_counts = {"dense_rescues_sparse": 1, "sparse_failure": 3}`

Per-synthetic cache quality:

- `synthetic/000000.png`: `sparse_failure`, sparse TE `10390.70cm`, dense TE `11492.94cm`
- `synthetic/000001.png`: `sparse_failure`, sparse TE `2398.01cm`, dense TE `3456.94cm`
- `synthetic/000002.png`: `sparse_failure`, sparse TE `1548.96cm`, dense TE `1657.80cm`
- `synthetic/000003.png`: `dense_rescues_sparse`, sparse TE `3891.31cm`, dense TE `2600.66cm`

Teacher-pose quality filtering with `max_sparse_te = 100cm` and `max_dense_te = 100cm` rejects all 4 synthetic records:

- before: `{"synthetic_rgb:accepted": 4}`
- after: `{}`

### Conclusion

The new RGB-only teacher path is operational and faster with 960px staging. It can generate non-empty WildGaussians synthetic RGB and write correct pseudo-query metadata, but the 500-step RGB teacher is still not strong enough to produce synthetic queries that survive the full STDLoc teacher cache quality gate. The important closure is that bad synthetic data is now stopped at a second gate: artifact score is no longer the only criterion.

### ShopFacade 960px WildGaussians 2000-Step Follow-Up

Training output:

- `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg960_2000step`
- Checkpoint:
  - `/mnt/pool/sqy/stdloc_la_rgb_teacher_smoke/ShopFacade_wg960_2000step/checkpoint-2000`
- Metrics:
  - final `train/psnr = 15.1364`
  - single train render eval `psnr = 14.7999`
  - single test render eval `psnr = 13.3393`

Synthetic RGB smoke:

- Manifest: `/tmp/la_pseudo_shopfacade_wg960_2000_render_smoke.jsonl`
- `train_rgb:accepted = 231`
- `synthetic_rgb:accepted = 4`
- All 4 WildGaussians renders were `960x540` with `render_scale = 0.5`.
- RGB statistics were no longer near-uniform:
  - per-channel std ranged from about `50.49` to `62.27` on 8-bit RGB.

Teacher cache on these 4 synthetic records:

- Cache: `/tmp/la_pseudo_shopfacade_wg960_2000_teacher_cache.pt`
- `stage_counts = {"dense_rescues_sparse": 2, "mixed_or_uncertain": 2}`

Per-synthetic cache quality:

- `synthetic/000000.png`: `mixed_or_uncertain`, sparse TE `5.99cm`, dense TE `3.64cm`, usable under 100cm gate.
- `synthetic/000001.png`: `dense_rescues_sparse`, sparse TE `7399.62cm`, dense TE `3884.08cm`, rejected by 100cm gate.
- `synthetic/000002.png`: `mixed_or_uncertain`, sparse TE `11.83cm`, dense TE `14.25cm`, usable under 100cm gate.
- `synthetic/000003.png`: `dense_rescues_sparse`, sparse TE `24.17cm`, dense TE `4.21cm`, usable under 100cm gate.

This is the first positive synthetic RGB signal in the new path: increasing the RGB teacher from 500 to 2000 steps changed the synthetic teacher-cache result from `0/4` usable to `3/4` usable under the same `max_sparse_te = 100cm`, `max_dense_te = 100cm` filter.

Real train RGB teacher-cache smoke:

- Cache: `/tmp/la_pseudo_shopfacade_wg960_2000_train_teacher_cache.pt`
- `stage_counts = {"teacher_ok": 4}`
- Sparse TE range: `1.49cm` to `3.47cm`.
- Dense TE range: `0.64cm` to `2.73cm`.

Mixed cache gate:

- Combined cache: `/tmp/la_pseudo_shopfacade_wg960_2000_mixed_teacher_cache.pt`
- Before cache gate: `{"synthetic_rgb:accepted": 4, "train_rgb:accepted": 231}`
- Smoke cache items available: 8.
- After 100cm cache gate: `{"synthetic_rgb:accepted": 3, "train_rgb:accepted": 4}`

Student training smoke:

- Temporary model: `/tmp/stdloc_la_pseudo_train_smoke_shopfacade_2000`
- Loaded baseline iteration: `30000`
- Ran to iteration: `30001`
- The training script printed the expected pseudo-query pool:
  - `after={"synthetic_rgb:accepted": 3, "train_rgb:accepted": 4}`
  - `real_weight = 2.0`
  - `synthetic_weight = 1.0`
- Final log:
  - `[ITER 30001] Saving LA Gaussians`
  - `[ITER 30001] base 0.150994 loc 0.602519 psnr 19.311`
  - `LA-STDLoc training complete.`

This closes the first end-to-end smoke for the new plan: RGB teacher map, synthetic RGB render, artifact gate, full STDLoc teacher cache, teacher-pose quality gate, mixed pseudo-query sampler, and student training all run in one consistent flow without using official test images as training data.

### Verification

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_la_artifacts \
  tests.test_episode_sampler \
  tests.test_render_artifact_weights \
  tests.test_teacher_stage_diagnostics \
  tests.test_full_script_args
```

Result:

- `Ran 72 tests`
- `OK`

Also passed:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  la_artifacts/*.py \
  scripts/prepare_rgb_teacher_manifest.py \
  scripts/prepare_nerfbaselines_colmap_dataset.py \
  scripts/build_pseudo_query_manifest.py \
  scripts/build_pseudo_teacher_cache.py \
  scripts/repair_render_artifacts.py \
  train_locaware.py \
  localization_training/episode_sampler.py \
  gaussian_renderer/__init__.py
```

and:

```bash
bash -n scripts/run_la_pseudo_query_pipeline.sh
git diff --check
```

## Full-Run Closure: ShopFacade and OldHospital

This follow-up used the preprocessed Cambridge data under `/mnt/pool/sqy/Cambridge_stdloc` and did not use official test images for pseudo-query training. Official test images were used only by `stdloc.py --sparse_only` evaluation.

### Full Pseudo-Query Cache

ShopFacade full pseudo-query run:

- Output root: `/mnt/pool/sqy/stdloc_la_pseudo_query_full_v1/ShopFacade`
- Manifest: `train_rgb:accepted = 231`, `synthetic_rgb:accepted = 32`
- Teacher cache items: `263`
- Stage counts: `teacher_ok = 185`, `mixed_or_uncertain = 48`, `dense_rescues_sparse = 21`, `sparse_failure = 8`, `dense_improves_sparse = 1`
- 100cm sparse/dense teacher gate: `train_rgb = 229/231`, `synthetic_rgb = 19/32`

ShopFacade low-detail artifact ablation:

- Output root: `/mnt/pool/sqy/stdloc_la_pseudo_query_lowdetail_v1/ShopFacade`
- Manifest after detector fix: `train_rgb:accepted = 231`, `synthetic_rgb:accepted = 15`, `synthetic_rgb:rejected = 17`
- Teacher cache items: `246`
- Stage counts: `teacher_ok = 185`, `mixed_or_uncertain = 46`, `dense_rescues_sparse = 11`, `sparse_failure = 3`, `dense_improves_sparse = 1`
- 100cm sparse/dense teacher gate: `train_rgb = 229/231`, `synthetic_rgb = 11/15`
- The top rejected synthetic renders were the same low-detail/blurred samples that previously produced very large teacher errors.

OldHospital full pseudo-query run:

- Output root: `/mnt/pool/sqy/stdloc_la_pseudo_query_full_v1/OldHospital`
- Manifest: `train_rgb:accepted = 895`, `synthetic_rgb:rejected = 32`
- All 32 synthetic renders had `artifact_score = 1.0`; visual inspection shows nearly black WildGaussians renders.
- Teacher cache items: `895`
- Stage counts: `teacher_ok = 103`, `mixed_or_uncertain = 330`, `dense_rescues_sparse = 237`, `dense_improves_sparse = 136`, `sparse_failure = 78`, `dense_regression_after_good_sparse = 11`
- 100cm sparse/dense teacher gate: `train_rgb = 877/895`

### Official Sparse-Only Evaluation

All values below are official test sparse-only pose evaluation from `stdloc.py --sparse_only`.

| Scene | Setting | Iter | Median TE cm | Median AE deg | Recall 5cm/5deg | Recall 2cm/2deg | Avg inliers |
|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade | baseline | 30000 | 3.349951 | 0.166517 | 0.728155 | 0.262136 | 388.107 |
| ShopFacade | train RGB only | 30100 | 3.183114 | 0.160812 | 0.747573 | 0.233010 | 429.126 |
| ShopFacade | train RGB + synthetic RGB | 30100 | 3.120151 | 0.156575 | 0.747573 | 0.233010 | 424.689 |
| ShopFacade | low-detail filtered train+synthetic | 30100 | 3.149924 | 0.164289 | 0.757282 | 0.242718 | 426.252 |
| ShopFacade | train RGB only | 30500 | 3.231896 | 0.156218 | 0.747573 | 0.291262 | 490.456 |
| ShopFacade | train RGB + synthetic RGB | 30500 | 2.964785 | 0.148777 | 0.757282 | 0.262136 | 468.777 |
| OldHospital | baseline | 30000 | 18.394085 | 0.338004 | 0.032967 | 0.005495 | 274.808 |
| OldHospital | all train RGB | 30100 | 19.699221 | 0.354685 | 0.038462 | 0.005495 | 274.632 |
| OldHospital | all train RGB | 30500 | 19.378080 | 0.359615 | 0.032967 | 0.000000 | 269.418 |

ShopFacade is the positive result: the 500-step train+synthetic run improves median TE from `3.35cm` to `2.96cm`, median AE from `0.1665deg` to `0.1488deg`, and recall@5cm from `72.8%` to `75.7%`. The train-only 500 run increases inliers and recall@2cm but does not improve median TE, so the accepted synthetic RGB contributes useful supervision on this scene.

OldHospital is not positive yet. The current run is effectively all-train-only because every synthetic RGB candidate was rejected. The method does not beat baseline there, and the failure is localized to two issues: unusable WildGaussians synthetic renders and noisy/heterogeneous train teacher stages.

### Visual Checks

Generated visual summaries:

- ShopFacade full: `/mnt/pool/sqy/stdloc_la_pseudo_query_full_v1/ShopFacade/visuals/visual_summary.json`
- ShopFacade full contact sheets:
  - `/mnt/pool/sqy/stdloc_la_pseudo_query_full_v1/ShopFacade/visuals/contact_sheet_train_rgb_accepted_dense_te_desc.jpg`
  - `/mnt/pool/sqy/stdloc_la_pseudo_query_full_v1/ShopFacade/visuals/contact_sheet_synthetic_rgb_accepted_dense_te_desc.jpg`
- ShopFacade low-detail ablation:
  - `/mnt/pool/sqy/stdloc_la_pseudo_query_lowdetail_v1/ShopFacade/visuals/visual_summary.json`
  - `/mnt/pool/sqy/stdloc_la_pseudo_query_lowdetail_v1/ShopFacade/visuals/contact_sheet_synthetic_rgb_all_artifact_desc.jpg`
- OldHospital:
  - `/mnt/pool/sqy/stdloc_la_pseudo_query_full_v1/OldHospital/visuals/visual_summary.json`
  - `/mnt/pool/sqy/stdloc_la_pseudo_query_full_v1/OldHospital/visuals/contact_sheet_train_rgb_all_artifact_desc.jpg`
  - `/mnt/pool/sqy/stdloc_la_pseudo_query_full_v1/OldHospital/visuals/contact_sheet_synthetic_rgb_all_artifact_desc.jpg`

Each contact sheet shows the query/render, nearest or source train RGB, artifact score heatmap, low-detail risk map, sparse/dense TE, stage label, artifact score, and repair action. The OldHospital synthetic sheet is the clearest qualitative failure: synthetic RGB is nearly black and the low-detail/artifact maps correctly flag it.

### Operational Notes

- GPU0/GPU1 completed the full runs. GPU2 still shows an 18GB stale `[Not Found]` context in `nvidia-smi`; `nvidia-smi --gpu-reset -i 2` returned `Unknown Error`, so clearing it likely requires host-level intervention or reboot.
- A reproducible environment failure was found and closed: new training processes could pick `/root/miniconda3/envs/iclpose/bin/nvcc`, causing `cuda_runtime.h: No such file or directory` during `gsplat` JIT compilation. Re-running with `CUDA_HOME=/usr/local/cuda` and `PATH=/usr/local/cuda/bin:$PATH` fixed the issue and rebuilt `~/.cache/torch_extensions/py38_cu118/gsplat_cuda/gsplat_cuda.so`.

### Current Conclusion

The RGB Teacher -> artifact gate -> pseudo-query teacher cache -> student training loop is now operational and gives positive official sparse-only support on ShopFacade. It is not yet scene-robust: OldHospital is blocked by bad synthetic RGB teacher renders and weak/noisy teacher-cache supervision. The next high-value work is to fix RGB teacher rendering for OldHospital, then rerun synthetic acceptance and student training before changing the LA student objective again.
