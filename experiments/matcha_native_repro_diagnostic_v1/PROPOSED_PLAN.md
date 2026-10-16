# Proposed native reproducibility diagnosis, not GPU permission

## Why this is needed
Completed original A used v1 pure observation code and full observations after each of1..8. B used unchanged numerical trainer/observer, a diagnostic-only CPU thread guard and only endpoint full8. B stopped at light iteration3 after exact comparison failed: L1 differs1 ULP, DSSIM and total loss2 ULP. Steps1/2 light records match exactly; view, ordered sampling stacks, RNG digest, LRs, cadence/phase, topology and all other iteration3 components match. This does not establish harmlessness, exact equivalence or attribution. No tolerance changes and no continuation are permitted by this proposal.

A and B producer-verified static state archives have identical14,461,905,798-byte file content SHA626f54752c06e2dff48274f371393499d84d2523338f14a1b80ba579486f2197. All actual computed prior/camera/mask caches in that archive match. No second large-state read was performed in this diagnosis.

## Minimal controlled comparison
Propose two new fresh, independent seed0 short jobs U1 and U2 on the same assigned physicalGPU0/logical0. Both use original v1 train_with_charts_observer.py and exact_resume.py without importing v2 diagnostic_cpu or changing any v1 byte. Preserve native horizon30000 and all frozen input/science/environment. Both observe light1..8 and full8 only, and stop at the original safe post-optimizer boundary8. Do not attach a reference that causes early exit; the diagnostic pair must retain both complete8 trajectories and endpoint evidence. Compare offline after both complete.

This pair holds the observer frequency fixed while removing the diagnostic guard. One new unguarded run compared only with historical A would confound the difference between full-every-step and endpoint-only observation schedules. Historical A remains an auxiliary exact comparison, but cannot replace the controlled U1/U2 pair when attributing the effect of the new guard.

U1/U2 execution must be individually reviewed, freshly named and launched by the parent. No automated retries or long-prefix promotion. The existing B/prefix/resume gates remain closed. If U1/U2 differ under identical original observer code, that establishes failure of native bitwise reproducibility for this tested configuration. It does not quantify an acceptable error or prove any checkpoint resume correct. If U1/U2 match, the guard remains a candidate, but a single matching pair cannot establish that rare CUDA nondeterminism is absent. Any subsequent controlled guard-on comparison requires a new reviewed plan; do not rerun B automatically.

## Evidence and comparison fields
Both jobs retain exact scalar byte encodings for all light loss components, views, cadence/phase, both sampling stacks, LRs, events, topology and all RNGs. Endpoint8 retains full model/Adam bindings and state, densification accumulators, actual mip, scheduler next LRs,7 EMAs, runtime/source/input binding and computed static caches. Do not drop states to save time.

Use original typed semantic digest for the complete comparison endpoint and preserve raw endpoint manifests and traces. Compute first-difference paths and loss ULP/absolute differences as descriptive diagnostics, with zero-tolerance verdict unchanged. A full typed digest difference is failure; a tolerance is never substituted retroactively. Larger statistical/numerical reproducibility claims would require an independently preregistered protocol and additional native calibration, approved separately.

The proposed thin launcher records actual DriverAPI UUID, visible device count, torch/Python/NumPy/CUDA/cuDNN versions, TF32/determinism/cuDNN flags, autocast status and native thread settings before and after. Endpoint8's original helper runtime binding supplies canonical after-native-import settings. Compare that binding field-by-field independently of the numerical endpoint digest. Any context mismatch is an invalid control, not a numerical pass or failure. Do not enable deterministic modes, TF32 changes, AMP or CPU thread limits.

## Actual available runtime evidence
A original endpoint pickle metadata, read through zipfile/pickletools without loading tensor storages, records Python3.9.12, Torch2.0.1+cu118, NumPy1.26.4, CUDA11.8, cuDNN8700, RTX3090, intra-op32/inter-op16; deterministic=false, warn_only=false, cuDNN benchmark=false, cuDNN deterministic=false, cuDNN TF32=true, matmul TF32=false. OMP/MKL/CUBLAS/PYTHONHASHSEED unset. Allocator max_split_size_mb:128.

B's actual initial diagnostic phase records snapshot_static8.860899s, diagnostic_digest12.494418s and atomic_save117.974503s. All say completed=true and native32/inter-op16 restored; guard verification requires identical RNG bytes and unchanged inter-op. Native trainer/observer bytes are identical A/B and contain no autocast or GradScaler calls. A/B launch environment dictionaries and actual context UUID before/after agree. The guard only sets intra-op count temporarily and has no TF32/cuDNN/determinism setter.

B failed before endpoint8 and therefore lacks a saved full runtime binding. Source/launch equality and guard restoration are evidence, but are not an actual saved B TF32/cuDNN measurement. Do not claim that missing measurement exists.

The native surfel rasterizer source contains15 float atomicAdd accumulations in cuda_rasterizer/backward.cu, including per-Gaussian color, transform, normal and projected-mean gradients. These are a concrete potential order-dependent native computation, not proof of the observed cause. The compiled operator SHA was frozen and identical across A/B.

## Cost and resources
Per unguarded original-helper run, actual A decomposition is528.360s setup + up to178.014s observed full capture +20.561s dynamic publication + about3s native/light work and4s outer launch overhead, approximately734s. The native32 static clone/hash CPU benchmark175.490s explains why endpoint-only unguarded observation can remain expensive. Use a900s forecast per run with explicit allowances600s setup +220s full endpoint +30s publication +50s other checks/margin. This is an estimate, not a new measurement.

Two sequential jobs therefore forecast1800s total elapsed (30 minutes), each with a proposed1200s hard outer deadline,120s cleanup reserve, no long continuation. The parent must approve these bounds before execution. Shared-host contention and I/O can exceed the estimate; a timeout fails closed and is not retried without review.

Each original-helper run independently computes and atomically publishes its14.462GB static archive and about3.997GB dynamic8 archive:18.459GB permanent state per run,36.917GB for two plus small logs. ExistingA18.459GB and failedB14.462GB remain preserved. No manual static cache reuse is introduced in this control. Temporary serialization payload is at most14.462GB and atomic rename does not permanently duplicate it. Peak observedA/B RSS approximately40.9GB; require at least96GiB available host RAM and64GiB free disk at each launch. ActualGPU0 memory at completion of failedB returned to3MiB. Original visible-device/lease/critical-launch/group-drain protections must be reused.

## Stop and handoff
Current B official Runner wc_job_U3bsvcWM_HwiBwWy is terminal failed/exit1, elapsed369.793s. GPU0 was observed idle/3MiB; parentPID gone. No8/14999/15101 model checkpoint or promotion receipt exists. Preserve all evidence and the independent FAIL_EXACT_AA receipt. This document authorizes no GPU execution and does not reopen any production gate.
