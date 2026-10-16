# Historical broad design: superseded, do not execute

Retained for audit. Current scope and execution gates are in README.md and LIMITED_14999_PLAN.json.

# Candidate exact MAtCha continuation extension

Status: implementation and micro CPU contracts only. GPU equivalence has NOT run. Corrected inputs, their independent receipt, and a production binding manifest do not yet exist. Nothing here authorizes GPU execution or acceptance of a final prior.

## Scope and immutable originals

- Original source: /root/MAtCha/2d-gaussian-splatting/train_with_charts.py, SHA256 5c5c7d924675dc3ad743208b3e2ca35d3d064f30892338845ff2c08a30b3a630.
- Original Gaussian implementation SHA256 8e69f8adc84faf148f3ed3f9a1b82e8e9931373c1f98478852824df7adc514b4.
- Frozen repair protocol SHA256 17f7d8bfb8dbdc522e22352d9d0055f014368dd190a004dc0ce6ae73088019c1; preparation producer SHA256 85bfab6bf42597ebb67c2a97437020e518ff661886da156be8366377c026df09.
- All additions are in this isolated directory. No original source, configuration, input, optimizer target, horizon, or camera-selection policy is edited.
- build_trainer.py creates train_with_charts_exact.py and trainer.diff from uniquely anchored, fixed-hash original text. build_observer.py creates train_with_charts_observer.py and observer.diff. The observer has no restore/skip branches and rejects --exact_resume.
- The observation-only reference keeps the native math/order but adds read-only state capture and a legal safe stop. It is an instrumented native reference, not a claim that the original uninstrumented executable wrote these traces.

## Boundary and state

The added boundary is after original lines 1943–1958, including optimizer.step, zero_grad and any requested native checkpoint. Densify/prune, opacity reset and scheduled mip refresh precede it unchanged. Original early_stop_iteration at 1914 is forbidden. At original iteration 30000 the optimizer step is skipped; a resumable checkpoint is never emitted for that terminal iteration.

Schema: matcha_exact_resume_v1. The payload strictly requires phase=after_optimizer_and_zero_grad, completed_iteration in 1..29999, next_iteration=completed+1, source/input/config/runtime binding, static reference, model, Adam, loop, RNG, emitted events, pipeline/debug and next-step LRs.

Captured state:
- All six trainable Gaussian tensors, dtype/device/shape/stride/row order, active/max SH.
- Named Adam parameter groups and identity bindings; all six step/moment states and group hyperparameters; zero gradients required at capture.
- max_radii2D, xyz_gradient_accum, denom, percent_dense, spatial_lr_scale.
- Actual use_mip_filter and actual mip_filter, without recomputing it on resume.
- Python, NumPy, Torch CPU and all visible CUDA RNGs. All devices synchronize before capture. RNG restores LAST after reconstruction, before the next native iteration.
- Both sampling stacks, preserving order, None and empty list; chart phase and mutable cadence; seven EMAs.
- Original 30000-step optimization configuration, scheduler construction, non-position base LRs/decay controls, current group LRs and next-step LRs.
- Already emitted evaluation/save/native-checkpoint events, and pipeline/debug flags.
- Computed static priors (depths, prior depths, confidences, normals, curvatures, weights, late weights and ordered pools), dense depth predictions, background and scene scale. Static bytes are saved once and referenced by path/SHA/size in every checkpoint.
- Camera ordering, complete rendering-camera tensor/metadata fields, RGB/alpha arrays, and raw mask lookup content are saved. Reconstruction of these is required to match saved values bitwise; computed priors restore the saved values. Dense network inference and initial mip computation are skipped on resume; chart-prior recomputation currently still occurs but its caches are replaced with the saved exact caches before training. This is intentional minimal instrumentation, with a startup-cost consequence.
- Full parsed scientific arguments must equal an independently frozen manifest. The manifest binds source, repaired inputs, RGB links, original point inputs, masks, network checkpoint, local compiled operators and independent input review. Runtime binds Torch/NumPy/Python, CUDA/cuDNN, visible GPU identity, backend numeric and CPU-thread flags.

Appearance, transient and color-correction modes are rejected. No exposure optimizer or AMP scaler is invented. Live network GUI and native warm resume are unsupported and fail closed.

Atomic publication: write a fresh hidden directory, fsync state, SHA, deserialize/readback and bitwise compare, write/fsync manifest, re-hash, fsync directory, atomic rename and fsync parent. No existing checkpoint is overwritten. Load verifies manifest/schema/size/SHA, bindings and state contracts before taking a training step. These are trusted local artifacts, not an untrusted-pickle import facility. Unpublished failed temporary directories remain for diagnosis.

## What ran

Micro CPU test: cybersim_agent Python, CUDA_VISIBLE_DEVICES empty, OMP/MKL/Torch threads=1. The current micro CPU suite includes controller integration and negative reference-coverage/source-binding contracts; consult cpu_validation.json for the final observed count/time. No production input arrays were opened and no CUDA was initialized.

Tests cover:
1. Parameter/Adam binding and exact next-step replay with Python/NumPy/Torch RNG.
2. Original native scheduler method evaluated at authentic iteration numbers through 30000.
3. Atomic/readback invariance and no RNG consumption.
4. Strided tensors, NaNs, signed zero and exact bytes.
5. None/empty/order sampling preservation.
6. Actual native sampling block replay through refill and cadence transition.
7. Wrong version/phase/terminal/next iteration, runtime binding, missing mip/denom/stacks/Adam/RNG, bad optimizer identity, and invalid events fail closed.
8. Changed scheduler configuration and unexpected gradients rejected.
9. Truncated bytes rejected before deserialization.
10. Controller tiny static-save/boundary/load/cache-restore flow.
11. Native argparse science recipe frozen without importing the trainer.
12. Native source SHA unchanged and CUDA uninitialized.

The CPU fixture has THREE Gaussians and 2x2 image tensors. Native training_setup is extracted and ONLY its two device='cuda' allocations are replaced with CPU for the fixture. Native update_learning_rate and camera-sampling statements are executed directly. These tests do not validate GPU rasterizer nondeterminism, native topology-changing CUDA kernels, real-data startup, real checkpoint throughput, or full-size memory.

## Proposed GPU plan, pending parent approval

All trials retain iterations=30000, native seed=0, the exact chosen original chart20+dense512 recipe, schedules, masks, numeric flags and GPU identity. No accelerated schedules or reduced point pools.

Cost-optimized proposal replaces the earlier 14-cut/N32 sketch:
1. Native observation-only A/A trial A stops safely at N=8; full state digests on steps 1–8. This covers the native first-step cadence mutation and first chart selection at step 5. It establishes initial native repeatability only.
2. Independent fresh native observation-only A/A trial B uses the same seed and inputs. Its steps 1–8 compare to A automatically and fail on the first mismatch. Only after this gate passes may the SAME continuous reference job continue to N=15101, subject to the separately approved wall/RAM/I/O budget. This avoids a third startup or restarting the prefix from zero.
3. Reference snapshots: 1, 599, 600, 999, 2999, 3000, 7000, 14999, 15000, 15099, 15101. Every snapshot is after the real native safe boundary. Complete comparison windows: 1:8, 599:602, 999:1002, 2999:3002, 7000:7003, 14999:15002, 15099:15101 (31 distinct steps). Other steps have lightweight camera pool/index/name, cadence-before/after, chart phase, all LRs, loss, point count and SH. Do NOT call this all-step full-state equality.
4. Separate short resume branches:
   - 1 -> 3: initial cadence/sampling and ordinary update.
   - 599 -> 602: actual first densify/prune+mip at 600.
   - 600 -> 602: restoration of the already changed topology/mip/Adam.
   - 999 -> 1002: SH increment at 1000 plus actual densify/mip.
   - 2999 -> 3002: actual SH+density+mip+opacity reset at 3000; mutable default dense cadence switches after camera selection at 3001.
   - 3000 -> 3002: resume already-reset optimizer/opacity and actual cadence crossing.
   - 7000 -> 7003: normal-consistency activation and dense regularization schedule change at 7001.
   - 14999 -> 15002: densification cutoff at 15000 (no densify/reset at 15000 because the guard is <15000), then dense schedule change at 15001.
   - 15000 -> 15002: restore across that schedule transition.
   - 15099 -> 15101: authentic first post-densification periodic mip refresh at 15100.
   Each branch compares every replayed step's lightweight evidence and full model/Adam/mip/loop/RNG/events/next-LR digest to the reference. A checkpoint saved exactly after an event must be tested as well as a cut immediately before it.
5. Step15101 is retained as the prospective formal prefix checkpoint. The parent may continue it to30000 only after independent input review, source review, native A/A and ALL proposed resume comparisons pass, with exact scientific configuration, static-cache/source/runtime hashes, canonical output provenance and no intervening changes. No need to throw away a validated reference prefix and retrain it. A prefix from a failed or incomplete validation must not be promoted.

No tolerance is accepted as proof of exact continuation: discrete states, RNG, topology and full-byte digests must match exactly; numeric acceptance tolerance is zero. Native A/A must be assessed before any resumed comparison. If initial A/A differs, stop before step9 and report stochastic GPU behavior; do not increase tolerances after seeing a mismatch. If only a later event neighborhood differs, stop that branch, report the exact discrepancy and distinguish unestablished native GPU reproducibility from a restore bug. A separate diagnostic repeat requires parent scheduling; approximate agreement cannot be relabeled exact. Initial A/A success alone is NOT a proof that every later CUDA event is deterministic.

The helper can compare against --exact_reference_trace, and raises at the first recorded mismatch. For a resumed comparison, the helper requires both lightweight records and full digests for EVERY reference step from next_iteration through exact_stop, automatically computes those digests, and rejects empty/missing coverage before scene initialization. The parent must still verify branch completion through its declared terminal step; trace-file presence alone is not a PASS receipt. No GPU driver/launcher/acceptance receipt is supplied here.

## Resource budget and cap gate

These are estimates/formulas, not throughput measurements. The historical native log reports 4010 seconds optimization and 4297 total, implying 287 seconds of startup/other work. Linear estimate for15101 optimization is 2018.5 seconds, or 2305.5 including that historical remainder. N8 contributes about1.1 seconds optimization, but startup still dominates. Actual native step cost/topology, corrected inputs, GPU, storage and warm caches can change these numbers.

Dynamic checkpoint at N points:
- SH3 model has58 FP32 values/point =232 bytes.
- Adam first+second moments add464 bytes.
- max_radii2D + xyz_gradient_accum + denom + actual mip add16 bytes.
- C(N) approximately712*N bytes, plus small scalar/group/RNG/serialization overhead.
- At5,613,128 initial points: C about3,996,547,136 bytes (approximately4.00GB decimal/3.72GiB). Recompute from actual tensor nbytes because densification changes N and shapes.
- Eleven reference snapshots about44GB retained at that size. The initial A/A terminal8 snapshot and ten branch terminal snapshots bring total to about88GB if all retained. Native PLY/evaluation outputs are additional.
- Each atomic publication performs approximately one state write and three complete reads (first SHA, deserialization readback, second SHA), or4*C I/O. Eleven reference snapshots alone imply about176GB filesystem traffic. All22 snapshot publications imply about352GB. A cold branch loads/checks dynamic and static state with about2*(C+S) reads before reconstruction.
- Full-state digests in31 reference comparison steps copy approximately124GB from GPU to CPU cumulatively and hash that data; checkpoint steps reuse that captured state. TrialA adds8 captures (~32GB); replay branches add25 captures (~100GB). This is cumulative bandwidth, not resident memory. No full tensor hashing on the other15070 steps.

Static size S must be measured at approved initialization, with no assumption that dense resolution equals the chart288x512 source grid:
- Let P_c be sum of final chart training pixels, P_d sum of dense training pixels.
- Camera RGB:12*(P_c+P_d) bytes, plus any alpha masks and small camera matrices.
- Dense depth cache:4*P_d bytes.
- Core chart depth/prior-depth/confidence/normal/curvature caches:28*P_c bytes.
- Active chart weights add up to4*P_c; optional late weights another4*P_c. Raw mask lookup arrays add their true nbytes.
- Thus typical no-late total S approximately44*P_c +16*P_d + masks/alpha, plus small metadata. If ALL532 views were288x512, the camera/dense/chart core would be about1.34GB before masks; this is only an illustration, not the observed dense size.
- Static bytes publish once per fresh native run; branches reference the same verified artifact, not copies.

Peak host-memory increment above native training: capture C plus readback C, at most one additional largest tensor temporary, plus live static/manifest state; restore similarly has loaded C plus equality-check C. Self.state/static large CPU copies are discarded after RNG restore. Use a conservative admission reserve >=2.5*C +2*S +4GiB ABOVE measured native peak; record process RSS. At C4GB/S1.34GB this is about17GB extra before native baseline. Do not launch on estimated free RAM alone if actual S/N are unavailable.

Per branch cost is NOT just2–3 optimization steps: it still initializes the native scene/Gaussians and recomputes chart priors, verifies/hashes all bound inputs, reads C+S, restores GPU tensors/Adam, compares camera bytes, captures replay digests, and publishes terminal state. Only dense network inference and initial mip computation are skipped. Until a measured resume startup R is available, use the conservative historical287-second remainder per branch plus read/hash/capture I/O. Ten such branches could cost roughly48 minutes of startup alone. These can share scheduling windows but must not concurrently contend with the parent's official timing or other scientific runs.

3500-second admission gate:
- Use Runner timeout at most3500 seconds, below the observed3600 cutoff; never detach.
- First approved N8 calibration must measure native startup B, C/S bytes, peak RSS, full capture/digest time H, publication time W, binding verification time V, and later one tiny resume startup R. Do not use a large artificial I/O benchmark.
- Reference-job forecast B+15101*measured_step_cost+31*H+11*W+V must be <3500 with at least120 seconds margin. This formula deliberately may double-count some overlap; conservative admission is desired.
- If forecast fails, do not start the long native reference. Parent must reduce validation I/O with a reviewed change or obtain an official execution path; cannot silently split an unvalidated reference, shorten30000, alter schedules or claim a failed job is continuous.
- Every branch budget R+replayed_steps*step_cost+replayed_steps*H+W+V must satisfy the same cap/margin.
- Root may choose an official compatible shorter safe boundary for future validated production chunks once exact continuation is established. Keeping the original30000 horizon is mandatory.

## Binding preparation and handoff

freeze_binding.py is deferred because it reads real assets and hashes a network checkpoint. It requires a real independent input review with schema=matcha_resume_input_review_v1, verdict=PASS and the exact training_plan_sha256. It creates matcha_exact_binding_v1 with complete parsed scientific arguments and file hashes. Do not synthesize that PASS review simply to get past the gate. The receipt schema is an integration proposal for the reviewer, not an already existing acceptance.

No files here are final-prior acceptance artifacts. Parent/root owns resource approval, independent input acceptance, exact-run comparison/coverage acceptance and promotion of a verified15101 prefix to the final30000 run.