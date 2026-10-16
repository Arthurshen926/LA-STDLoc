# Matched100 timing preparation

This package is isolated from the original accuracy, native sources, model training, old Any full-query timing and paper. CPU preparation is authorized. GPU execution is not authorized by any preparation command.

## Scientific design

- Five Cambridge scenes, 20 native cells (STD/ULF × mapping semantics off/on) and10 Any Full/Compact cells. Query semantics is on everywhere.
- Select100 names using the already approved SHA256 namespace/seed/scene/complete-name ranking. Execute in the frozen full-name order, identical for every method.
- Each of three rounds is a fresh process. It first executes50 distinct queries, the first50 of the fixed100. Those calls are excluded from measured statistics. Then execute the full100. No cached query features.
- The original all1918-query accuracy experiment remains separate. No old telemetry or extracted old Any subset is accepted as these new measurements.

## Timing boundaries

Native class and method ASTs remain unchanged. The existing wrapper's synchronized native core/coarse-prefix/dense-increment boundaries remain intact. STD uses dense1 and ULF dense5 from the exact original recipe. STD's added query semantic preprocessing is explicitly an external RGB-mask variant.

Both native Scene loaders originally retain lazy PIL images. For every new warmup and measured call, the adapter reopens and fully decodes the frozen RGB file, substitutes an equivalent fresh image in that query's metadata only while calling the original native SceneDataset.__getitem__, then restores the original metadata. CPU tests compare actual native pixels, camera tensors and metadata. No method numerical parameter or feature output is changed.

Native rgb_decode_ms is PIL open+load; camera_materialize_ms includes the unchanged native resize, tensor/camera construction and geometric/device work. mask_preprocess_ms includes synchronized conversion/composition/transfer/application; pure native mask_compose_ms is unavailable (null). Native frame_ms continuously spans fresh decode through core return, including native pre-core bookkeeping. It excludes post-core accuracy metrics and serialization. STD core receives no GT; its outer frame includes camera/GT metadata construction. ULF core/frame explicitly include native GT projection and inlier diagnostics.

Any uses the same accepted ColmapDataset and Localizer. Its rgb_decode_preprocess_ms, CPU mask_compose_ms, runtime/core and outer frame retain the old scientific operations, now rerun on the fixed100. Separate coarse/dense boundaries are unavailable, never fabricated as zero. Segmentation-network inference is excluded for all methods; masks are precomputed.

Initialization is separately recorded for each fresh round. Native mask pickle loads are separately recorded inside initialization. Any dataset initialization includes mask archive loading and is labelled accordingly; no standalone archive cost is invented. Interpreter startup and launch-time hash verification are outside initialization. Shared feature weights are separately reported and excluded from map storage.

Torch allocated/reserved peaks are captured before and after core because native ULF may reset its counters internally. Maxima over measured frames are separate from warmup and initialization. ru_maxrss is process-lifetime, including initialization, not measured-only memory or total device VRAM.

## CPU preparation and review

Use the Python3.9 g4splat environment for planning and launching. The launcher execs each native adapter using its frozen original Python3.8 environment; Any uses g4splat.

1. Run the CPU tests.
2. Run: python -m experiments.native_baseline_matched_benchmark.plan prepare
3. Independent review must bind the resulting protocol SHA with complete=true, verdict=PASS, scope=matched_timing_cpu and no_gpu_run_performed=true.
4. Only after all20 native accuracy contracts include accepted final models and full-query verification: run plan bind. This creates one immutable30-cell asset manifest and90 round contracts. Missing final assets/accuracy references abort before any partial binding. No placeholder model hashes. Old Any full-query round0 poses/success are bound as accuracy evidence only; its times are never reused.
5. Parent releases all training, accuracy and heavy CPU/I/O work, and supplies one exact round-contract-bound, protocol-bound physical GPU UUID lease. cpu_io_quiet and all_training_accuracy_finished must both be true.
6. Run each round serially as a separate official finite Runner process, timeout3000 seconds or below. No detached process, no multi-round controller, no1-hour-spanning launcher.
7. Observe the exact Runner terminal state and save exact case/round/contract-bound terminal.json, including canonical path/SHA/bytes references named result and launch; native jobs also require native_results and native_receipt references. Aggregate only after all90 distinct jobs completed successfully. Changes to result bytes, launch bytes or output paths invalidate acceptance.

No GPU run is part of this package's CPU acceptance.

## Isolation limits

The launcher rejects compute on every physical GPU, takes one kernel-released common advisory lock, and execs one worker. Initialization observes exactly one GPU process on the leased device; subsequent1s checks reject additional GPU processes anywhere. This NVML observation does not claim host-PID namespace ownership. At initialization and completion, actual CUDA device_count=1,current_device=0 and DriverAPI current-context UUID must match the lease, case, round and contract. Monitor shutdown joins the thread, checks its final error and performs a last scan; round.json is published only after clean shutdown. A parent-attested quiet-host release is required. CPU load/disk telemetry is retained for independent audit; this package does not automatically certify absence of every unrelated CPU/I/O task.

All measured and warmup rows retain pose/success/matches/inliers evidence where exposed (null means unavailable). Native results and receipts are separately hash-bound and checked by query name against each row and frozen GT. Cross-round/full-query pose differences and reported failures are shown for audit; they do not filter timing rows or alter RNG. Family-specific required timing fields, unavailable fields, memory values and native core=coarse+dense are validated.

Every output has exclusive creation semantics. A failed run's directory is preserved and cannot be overwritten. A rerun needs a separately reviewed attempt contract/output design rather than deleting evidence. The aggregate refuses missing/duplicate jobs, altered names/order/rounds, and invalid or unavailable-required timings.
