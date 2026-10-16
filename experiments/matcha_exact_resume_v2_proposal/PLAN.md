# Candidate v2 observation plan (not executable)

Status: proposal only. The running native N8 trial A and every file bound by its v1 manifest remain unchanged. No new GPU job, thread setting, or training modification is authorized by this file.

## Read-only diagnosis, 2026-10-12 07:11–07:15 UTC

Run: /mnt/pool/sqy/anygsloc_prior_full_compact_20261011_v1/matcha_exact_runs/native_n8_trial_a_stmary_20261012_v1
Existing Runner job: wc_job_EpbZHhHqGMseMNBP; child PID 17487.

- Trace showed light step 4 after complete full states 1–3, subsequently complete full states 1–5 and light step 6. The tqdm progress bar does not advance until its native ten-step cadence; it is not a reliable N8 liveness indicator.
- Actual initial Gaussians: 5,613,130. Dynamic model/Adam tensor bytes C=3,996,548,584. Static tensor bytes S=14,458,002,788; serialized static bytes=14,461,905,798.
- Fresh setup observation: 528.3602 seconds including static publication 134.5396 seconds. This is not purely native startup: it also includes unseparated initial static packing and digest work. The native log reaches masks-loaded at 07:01:19 UTC; absolute separation inferred from filesystem clocks is approximate.
- Completed full-capture durations at steps 1–5: 121.3157, 138.5464, 45.1858, 104.8193, 94.6392 seconds. Native-step plus lightweight-observation gaps between those records are about 0.26–0.37 seconds; these early gaps must not be extrapolated over all 30,000 steps.
- During incomplete step-4 full capture, static state.pt and manifest.json were already published, no temporary checkpoint directory existed, and every /proc/17487/io counter was unchanged over one second. Thus this interval was in-memory capture/hash/validation, not checkpoint disk serialization or readback.
- The code location is exact_resume.py lines 648–666: synchronize, model/Adam pack, actual static snapshot and digest, schema checks, then full dynamic-state digest. Dynamic disk publication occurs only at registered saves or stop (step 8 for trial A).
- Static data are 78.34% of C+S bytes sampled at every full boundary. This byte fraction is not a measured time fraction. Existing instrumentation cannot disaggregate dynamic copying, static copying and SHA time.
- validate_model and validate_checkpoint inspect descriptors/topology/bindings; they do not perform a bulk floating-point reduction. pack uses tensor.cpu().clone and NumPy copy. semantic_digest reads exact raw bytes; no lossy conversion or tolerance is used.
- The child had 100 OS threads and affinity CPUs 0–31, with no OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS setting. A one-second sample found the main thread and 31 later-created threads each using 82–100 CPU ticks. Loaded libraries include Torch libgomp and NumPy/OpenCV OpenBLAS. Attribution of the active group to Torch's OpenMP pool is an inference, not an attached native stack trace.
- Other two campaign lanes were doing CPU selector work concurrently. These timings are not an exclusive-CPU calibration. No process was attached, signaled, stopped or modified.

## Minimal required v2 delta: B endpoint-only A/A observation

Root's revised registration is:
1. Keep all original trial-A light/full records 1–8 and its step-8 checkpoint intact.
2. Trial B compares exact lightweight records for all steps 1–8 and the complete full state only at step 8. Zero tolerance; missing step/component/reference or first difference fails.
3. If this A/B observation comparison passes and separately measured resource admission permits it, the same B process continues to 15101, with full captures and checkpoint saves only at 14999 and 15101.
4. Resume only from 14999; require its full reconstructed state, light steps 15000–15101 and full endpoint15101 to match before same-process continuation to30000.
5. Claims are trajectory equality plus registered endpoint byte equality, not per-step complete weights.

Candidate implementation:
- Create a separate version directory; do not edit v1.
- Produce a strict, auditable projection of the completed raw A trace containing exactly light1..8 and full8. Record the raw trace path/SHA and projected path/SHA; do not modify the raw evidence.
- Narrow fresh-reference validation to exactly that registered coverage. It currently requires full1..8, so a new independently reviewed helper is necessary.
- Omit broad comparison windows in B. Automatic reference-triggered capture then fires only at8, plus explicit saves14999/15101.
- Keep the native trainer arithmetic, optimizer, original horizon30000, view sampling, schedules and immutable input recipe unchanged. Reproduce the instrumented trainer from the original fixed-SHA source and review the helper-only delta.
- Freeze a new B binding and explicitly bind the relationship between v1 A and v2 B. They have the same native scientific code/input recipe but different observation control source. The A receipt cannot silently stand in for v2 source review.

## Optional diagnostic CPU-thread guard; not yet implemented

Only consider this after independent review and CPU contracts. It is not necessary to change training thread settings to enact the endpoint-only plan.

A context manager could save torch.get_num_threads(), set a fixed small intra-op count for a wholly diagnostic block, and restore the exact prior value in finally. Do not change inter-op count, OMP/MKL environment, or native training operations.

Eligible blocks contain only read-only snapshot copies, raw-byte hashes/equality, serialization/readback and schema checks. Thread switching must not surround optimizer/backprop/render/loss/scheduler/sampling or model restoration/training_setup. Synchronize as already done at full boundaries; do not add synchronization within native numerical code.

Required proof before any GPU use:
- Bitwise identical packed tensor/NumPy descriptors and bytes, including NaN payloads, signed zero, noncontiguous strides and complete tiny model/Adam state.
- Same semantic digest and serialized readback equality.
- Python, NumPy, Torch CPU and visible-CUDA RNG checks around the actual diagnostic guard (GPU RNG only in later authorized tests).
- Exact native intra-op thread count before and after, including nested contexts and exceptions.
- Original numerical-source function hashes and scientific args unchanged.
- Preserve the same static bytes and all endpoint checks. No cache or Adam/mip omission.
- Add separate diagnostic phase timing for model pack, static pack, static SHA, dynamic SHA and atomic write/SHA/readback/byte-equality/publication without adding extra full reads.
- Independently review the new source and then require A-light1..8/B-light1..8 plus A-full8/B-full8 equality. No speedup is assumed in the budget before it is measured.

A single observation-level guard is preferable to decorating recursive pack/equal calls, which would repeatedly change thread pools. Existing native runtime binding must be captured with the original thread count and verified after restoration.

## Budget after endpoint-only B

B now has three full comparisons: 8,14999,15101 (previously ten). At initial C/S this removes seven captures, about129.18 GB of raw snapshot payload before duplicate memory reads, and preserves the A evidence. Static integrity checks remain at all retained endpoints.

Budget formulas, all from actual measurement when available:
- B/prefix = cold imports/preflight + fresh native/static setup + native numerical steps1..15101 + F8+F14999+F15101 + publication14999+publication15101 + native evaluation/save events + lightweight trace overhead + margin.
- Resume/finish = cold imports/preflight + read/verify dynamic and static artifacts + fresh camera reconstruction/static equality + model restore/readback + restored14999 full check + numerical steps15000..30000 + lightweight comparison15000..15101 + full15101 + native final evaluation/save/export + margin.
- The resume budget must use actual14999 topology, checkpoint size and measured restore/setup. N8 alone cannot admit it.
- C shrinks or grows with Gaussians; S need not shrink with topology. Do not scale all full-capture time by N.
- Both reviewed predicted totals must be below3380 seconds, with3500 total deadline and reserved cleanup. Do not launch on an estimate that merely hopes for a cutoff.

Historical budget evidence:
Use ONLY the formally bound /root/MAtCha/output_cambridge_full_retained_v2/StMarysChurch_n20_long_masked_retrain/logs/train_30000_driver.log, whose current SHA exactly matches the training plan: d695c4472d21b730b8af937203d555e080b4cdd0ab43a557dfab16b79fd8104e.
Its tqdm elapsed:1000=03:34 (3,590,314 points),3000=09:15 (2,940,275),7000=18:21 (305,437),10000=24:55 (298,736),14900=35:52 (460,299),15000=36:09 (2169 seconds),15100=36:26 (2186 seconds),15110=36:28,30000=1:06:50 (4010 seconds).
Thus the historical 15101 prefix center is about2186 seconds, and the subsequent native remainder about1824 seconds, before new checkpoint/trace overhead. These cumulative native progress times already include native evaluation/save interruptions before each observation; do not double-count those events.
This is not a measurement of corrected-input training. Native topology may differ and concurrent CPU workloads affect new observation costs. Resume admission still requires actual14999 evidence.
A second log under output_cambridge/runs/.../refine_free_gaussians.log (SHA0749a3d40d0ef7b2c85aa11cad491d733f380f756fac157f096dbb78ecf2e952) ends at2761 seconds but lacks the same formal recipe-binding evidence. It is excluded from all budget arithmetic; do not replace the formal4010-second baseline with it.

No v2 implementation, new binding, or GPU execution is performed by this proposal.
