# AnyGSLoc

AnyGSLoc is a source-image-free sparse visual localization pipeline built from a frozen RGB Gaussian prior. The paper mainline is intentionally small:

1. render RGB, alpha, and depth at mapping camera poses;
2. extract native SuperPoint observations from the rendered RGB;
3. apply the V2 render-quality filter before cross-view association;
4. build reciprocal, epipolar-consistent projective tracks;
5. triangulate robust 3D Anchors and aggregate one descriptor per track;
6. localize a real query with exact global cosine Top-1 and one standard PoseLib solve.

The optional `AnyGSLoc-R` arm adds a query-specific, sparse, map-read-only reserve refinement and at most one additional pose solve. Base is the primary paper method; `-R` is reported separately.

## Frozen scientific scope

The formal mainline does **not** use:

- source mapping RGB pixels;
- test queries during map construction;
- offline self-localization feedback or descriptor/metric training;
- query rendering or dense matching;
- scene recognition, learned result selection, or map writeback.

Mapping camera poses and intrinsics are required. Historical LaFGS feedback experiments remain in the repository for reproducibility, but no formal AnyGSLoc entrypoint imports or executes them.

The machine-readable contract is [configs/anygsloc_mainline.yaml](configs/anygsloc_mainline.yaml). Loading it fails closed if a forbidden mechanism is enabled.

## One scene

With a normalized prior manifest:

```bash
python -m scripts.run_anygsloc_scene \
  --scene KingsCollege_vanilla_3dgs \
  --dataset /mnt/pool/sqy/Cambridge_stdloc/KingsCollege \
  --prior-manifest /mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802/priors/KingsCollege/vanilla_3dgs/stdloc_model/rgb_prior_manifest.json \
  --output /mnt/pool/sqy/anygsloc_paper_experiments_20260902/CambridgePrior/KingsCollege_vanilla_3dgs \
  --gpu 0
```

Use `--dry-run` to print the complete command and provenance plan without creating outputs. The runner is restartable at completed artifact boundaries and rejects partial map/evaluation directories.

## Paper experiment matrix

The frozen matrix is [configs/anygsloc_experiments.json](configs/anygsloc_experiments.json). It contains:

- the primary Cambridge + 7Scenes + 12Scenes 24-scene evaluation;
- the reusable vanilla 2DGS / vanilla 3DGS / AnySplat robustness study;
- construction, capacity, mapping-camera, prior-quality, and online ablation definitions;
- accuracy, runtime, memory, map-size, and build-cost reporting fields.

Audit every input before launching work:

```bash
python -m scripts.run_anygsloc_matrix \
  --group primary_24_scene \
  --audit-only
```

Run the missing 24-scene cells on two GPUs while reusing the five already validated high-capacity Cambridge Base maps:

```bash
python -m scripts.run_anygsloc_matrix \
  --group primary_24_scene \
  --gpus 0,1 \
  --max-workers 2 \
  --reuse-existing-base
```

Run the six off-the-shelf prior cells:

```bash
python -m scripts.run_anygsloc_matrix \
  --group prior_robustness \
  --gpus 0,1 \
  --max-workers 2
```

The complete experimental protocol and promotion rules are in [docs/anygsloc_experiment_protocol.md](docs/anygsloc_experiment_protocol.md).

## Evaluation policy

Primary accuracy is median translation error and median rotation error, accompanied by p90/mean translation error, R5, and catastrophe count. Report mean/p50/p90 end-to-end latency, stage timing, second-solve rate, peak GPU memory, peak CPU RSS, map size, Anchor count, and map-build wall time.

All paper tables must bind the dataset, prior, map, identity metric, configuration, commit, seed, and result hashes. A result tuned on the same test queries must be labeled test-calibrated and cannot be presented as unseen generalization.
