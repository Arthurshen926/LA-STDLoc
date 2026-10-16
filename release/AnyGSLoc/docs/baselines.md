# Native STDLoc and ULF-Loc

The comparison keeps the two methods' native feature extractors, maps, solver budgets, and refinement stages. AnyGSLoc does not replace their frontends. `reproduction/baselines.py` is a small wrapper around the accepted native source snapshots. `native_hooks.py` adds controlled query masking and timing boundaries. AST transformations fail if expected native source anchors change.

## Environment and sources

The [environment launcher](environments.md) selects the existing `native` profile and applies the environment variables below automatically. Both comparison methods use that same profile.

The accepted native environment is Python 3.8, PyTorch 2.4.1+cu118, torchvision 0.19.1+cu118, NumPy 1.24.4, PoseLib 2.0.5. Additional installed versions are recorded in `reproduction/environment_ulfloc_repro.json`. Use each native repository's README and CUDA submodule installation commands. Keep this separate from the AnyGSLoc rendering environment. The source snapshots in `assets/third_party/{STDLoc,ULFLoc}` have their Python/CUDA/config file hashes recorded in `reproduction/native_versions.json`; those copied snapshots do not retain `.git`, so no unverified commit ID is supplied.

From the AnyGSLoc source checkout, expose the wrapper to that environment:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export ULFLOC_FUSION_BATCH_SIZE=1
export CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.6  # reference RTX 3090; change for another GPU
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Do not install the AnyGSLoc Python >=3.9 package into the Python 3.8 baseline environment. The wrapper and the small dataset/timing helpers it imports support the native environment directly from the source checkout.

## Required model inventories

Both methods require native `cfg_args` and `point_cloud/iteration_30000/point_cloud.ply`. STDLoc additionally requires `detector/sampled_idx.pkl` and `detector/30000_detector.pth`. ULF-Loc requires `test/keypoints_sampled_idx.pkl` and `test/keypoints_features.pkl`. These are scene-specific deployment assets and are included in paper storage. Shared pretrained extractors remain in the native dependency environment and are excluded from scene-map storage.

Use the accepted mapping-semantic-on and mapping-semantic-off models separately. Changing a flag while reusing the same trained model does not reproduce the mapping-mask ablation. St. Mary's Church must use the repaired current model inventory bound by `assets.json`, not a similarly named older directory.

## Accuracy

Activate the native environment, then run this command for each of five scenes, both methods, and both mapping-semantic variants:

```bash
python -m reproduction.baselines accuracy --method STDLoc \
  --repo assets/third_party/STDLoc \
  --dataset assets/datasets/Cambridge/ShopFacade \
  --model assets/native/STDLoc/on/ShopFacade \
  --config configs/native/stdloc_cambridge.yaml \
  --mapping-semantics on --scene ShopFacade --output runs/native/STDLoc/on/ShopFacade
```

For ULF-Loc use `--method ULFLoc`, `assets/third_party/ULFLoc`, its model path, and `configs/native/ulfloc_cambridge.yaml`. Add `--dry-run` to inspect the native argv without loading a model.

STDLoc uses the paper's controlled external RGB-mask preprocessing variant. ULF-Loc retains its native query mask branch. Query semantics are on for both mapping conditions. The wrapper creates an isolated model view from the supplied files and writes into a fresh output directory. It does not mutate the supplied models.

Native result rows contain `sparse_TE/sparse_AE` and `dense_TE/dense_AE`. Preserve all rows, including catastrophic errors and zero-inlier returned poses. The query order is the dataset test order and is saved in `manifest.json`, together with model hashes/storage. The initial and final rows share the same native deployment inventory; do not sum their storage twice. ULF-Loc's case-sensitive dataset-path dispatch is supported by a local lowercase `cambridge` symlink inside the output directory.

After all five scenes finish, run `python -m reproduction.report native --root runs/native/STDLoc/on --output runs/native/STDLoc/on/report.json`. Repeat for the other three method/mapping conditions.

## Timing

Repeat accuracy arguments with `timing` and a fresh `--output` for each `--round 0`, `1`, `2`. Always specify the scene. One round is one process. Run all jobs serially on an idle GPU. Each scene's fixed 100 queries and 50 warm-up calls match AnyGSLoc's selection.

```bash
python -m reproduction.baselines timing --method STDLoc \
  --repo assets/third_party/STDLoc --dataset assets/datasets/Cambridge/ShopFacade \
  --model assets/native/STDLoc/on/ShopFacade --config configs/native/stdloc_cambridge.yaml \
  --mapping-semantics on --scene ShopFacade --round 0 \
  --output runs/native_timing/STDLoc/on/ShopFacade/round0
```

The initial checkpoint is measured inside the complete call. ULF-Loc's native `sparse_only` segment excludes earlier dense-feature precomputation, so it differs from the cumulative initial checkpoint used in main T4. `round.json` retains both the stage timings and complete-frame/core measurements. Do not replace cumulative time with the smaller native sparse-only segment.

## Retraining

`python -m reproduction.baselines train ...` exposes the accepted wrapper's native training phase, using the same method/config/mapping-semantic arguments and a fresh output. Native STDLoc has a Gaussian/feature training phase followed by detector training; retain the native scripts and their required pretrained networks. The current reproducibility entry for the paper starts from the exact trained model inventories, which avoids silently substituting a new upstream SfM initialization. Training is not rerun as part of this source-cleanup validation. Record the initialization, training configuration, complete model inventory, and source versions if retraining; compare its accuracy as a new run rather than assuming bitwise equality with released models.
