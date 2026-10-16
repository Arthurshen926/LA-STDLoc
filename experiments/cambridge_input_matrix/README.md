# Cambridge input availability matrix (isolated extension)

This package is an opt-in research experiment. It does not change production
source-free contracts, existing maps, results, paper text, or Gaussian training.
The parent approved vanilla 3DGS as the candidate prior after an exploratory
Cambridge comparison. This is disclosed as exploratory selection, not held-out
model selection. All five scenes are predeclared. Seed and budgets never depend
on the new matrix's test outcomes.

## Scientific definition

Rows: pure Gaussian, original real mapping RGB, and Gaussian + real mapping RGB.
Columns: AnyGSLoc construction and a self-implemented, fixed-known-camera
SfM-style construction. The classic reference is not an official COLMAP pipeline.

The original real RGB is read from the audited same-K rectified `processed`
directory. The raw `seq*` images have distortion and must not be passed unchanged
to the centered-pinhole geometry. Mapping is exactly the non-test camera
whitelist, attested by the existing camera-registry implementation and image
hashes. No test RGB is opened by prepare, extract, build, or selection.

This is a computational-input guarantee, not a claim of zero test-metadata I/O.
The shared ColmapDataset loader reads the full camera registry and test-list text,
including test pose fields, before applying the mapping split. Query poses/GT are
not passed to association, triangulation, fusion, or selection. A CPU counterfactual
test changes test rotation/translation and GT-list columns while holding mapping
inputs fixed; all six constructions and spatial subsets remain exactly identical.

Both constructors receive the same complete unmasked SuperPoint feature pool.
There are at most 2048 retained observations per physical camera before V2.
Hybrid keeps stable score-top1024 real plus top1024 rendered observations;
shortages are not transferred/refilled. Both modalities share one camera record.
The full-available (up to 4096) hybrid pool size is reported separately. It does
not silently double the primary experiment's observation budget.

Any uses the existing cycle/chain association, robust ray triangulation, and
view-bin balanced robust descriptor fusion. Gaussian modes apply the existing
V2 certificate and depth-proposed completion; final XYZ remains ray triangulation.
The real-only Any row is explicitly "AnyGSLoc without Gaussian": V2, alpha/depth
evidence, and completion are disabled. The remaining association, ray and fusion
modules still differ from classic. No artificial difference is added to force
six distinct outcomes.

Classic uses ordinary mutual top1 descriptor/epipolar matches, confidence-sorted
conflict-free tracks, normalized DLT with fixed-camera point-only Huber fitting,
and arithmetic-mean descriptors. Its input adapter removes all auxiliary evidence
before numerical construction. It cannot construct points from Gaussian depth.

Every cell reports all valid landmarks (full) and a common spatially selected
map. Common K is computed only after all six full maps exist, as the minimum of
the predeclared reference K and the six attainable capacities. Exactly the same
0.5m voxel round-robin selector is used in every cell. Native Any compact
selection is outside this primary comparison, so it cannot disadvantage classic.

Queries always use the same frozen semantic-on frontend (`explicit_valid_mask`),
native scene calibration, PoseLib configuration, seed 2026, and exact test list.
Mask bytes are hash-bound. At evaluation only, a common query RGB/GT/K/calibration
snapshot is created and revalidated before and after every cell. Its SHA and
output hashes are bound into attestations. Historical semantic-off scores are not reused.

Preparation also audits native training argv, cfg_args, producer/input manifests,
actual prepared mapping image bytes and trainer camera registry. It checks each
initial point against its original complete mapping-only track, and verifies the
training initialization PLY coordinates. Evidence paths and SHA hashes are frozen.
OldHospital/KingsCollege use rectification_replay_manifest.json; the other three
use input_manifest.json. Missing or inconsistent evidence blocks preparation.

### Explicit differences from historical pure-GS runs

The controlled matrix extracts a common unmasked top2048 feature pool before
V2. Historical Any caches sometimes use alpha-gated detection/refill. Therefore
their maps and scores are not automatically compatible and this driver builds
all six cells afresh. The original numerical Any kernels are unchanged: three
isolated adapter copies only alter scope checks, schema, provenance, and the
raw-only branch. `kernel_adaptations.json` registers exact edits; CPU regression
tests compare original and adapted pure-GS association, triangulation, fusion,
and depth completion using the same inputs. Core matcher, triangulator and fusion
functions remain imports from production.

## Entry points

Use the repository checkout and g4splat Python:

    PY=/root/miniconda3/envs/g4splat/bin/python
    $PY -m experiments.cambridge_input_matrix.driver --help
    $PY -m experiments.cambridge_input_matrix.driver design
    $PY -m experiments.cambridge_input_matrix.driver check-kernels
    $PY -m pytest -q tests/test_cambridge_input_matrix.py tests/test_supplement_experiments.py

The example prior manifest points to the already-existing vanilla3DGS priors.
Preparation freezes all code, mapping-image, mask and prior hashes. It launches
nothing. Choose a NEW output directory; the package never overwrites results.

    OUT=/mnt/pool/sqy/anygsloc_input_matrix_vanilla3d_20261011
    $PY -m experiments.cambridge_input_matrix.driver prepare --root "$OUT" \
      --reference /mnt/pool/sqy/anygsloc_supplement_20261009/protocol.json \
      --prior-manifest experiments/cambridge_input_matrix/vanilla3d_priors.json

### GPU execution requires the parent's separate go-ahead

No GPU command below was executed during implementation. Recheck existing jobs
and GPU users first. Run a fixed ShopFacade smoke first; it was chosen as the
smallest scene, not from accuracy. Never run formal timing concurrently.

    $PY -m experiments.cambridge_input_matrix.driver extract --root "$OUT" --scene ShopFacade
    $PY -m experiments.cambridge_input_matrix.driver build --root "$OUT" --scene ShopFacade \
      --input-mode raw_only --method anygsloc --workers 4
    $PY -m experiments.cambridge_input_matrix.driver build --root "$OUT" --scene ShopFacade \
      --input-mode raw_only --method classic --workers 4

Then build the other four ShopFacade cells (pure_gs/hybrid x both methods).
Only after all six exist:

    $PY -m experiments.cambridge_input_matrix.driver select --root "$OUT" --scene ShopFacade
    $PY -m experiments.cambridge_input_matrix.driver evaluate --root "$OUT" --scene ShopFacade \
      --input-mode raw_only --method anygsloc --selection full
    $PY -m experiments.cambridge_input_matrix.driver evaluate --root "$OUT" --scene ShopFacade \
      --input-mode raw_only --method anygsloc --selection common_spatial

Repeat evaluation for every cell and selection, then the predeclared scene order:
ShopFacade, OldHospital, KingsCollege, GreatCourt, StMarysChurch.
Finally `report --root "$OUT"` requires all 60 evaluation cells; partial results
cannot silently become a complete report. Report uses scene-equal averages;
failure queries remain in metric denominators. Nonfinite numerical results are
serialized as null, not dropped.

Construction failures intentionally preserve their directory and fail closed.
Do not rerun into it or silently delete it: inspect the error, retain evidence,
then version a corrected root if the protocol/code must change. A recovery/resume
or durable queue can be added after review; this package does not launch a daemon.

## Resources and limitations

No new Gaussian training. One scene at a time; one GPU process at a time for
reproducible conditions. Budget a 24GB GPU, 4 CPU workers, at least 32GB RAM
(64GB preferred for large scenes), and 40–60GB free disk for fresh sparse caches,
construction evidence and full/common-K maps. Dense rendered RGB/depth/alpha
are used one camera at a time and are not stored in the new caches.

Historical five-scene timings: classic render/match/fit ~19 minutes; Any
association/reconstruction ~26 minutes; one five-scene evaluation ~5 minutes.
A fresh single-prior matrix is estimated at 2–4 GPU-machine hours plus CPU/I/O.
This is a planning estimate, not a benchmark; the new threaded classic fitting
path and V2-at-two-modalities extraction must be timed in the authorized smoke.
No native selector sweep or formal three-round warmed timing is included.

CPU tests are synthetic and prove contracts and numerical adapter equivalence.
They do not prove rasterizer operation, real-data completion yield, peak VRAM,
full-scene runtime, or accuracy. Those remain pending the separately authorized
GPU smoke and five-scene execution.
