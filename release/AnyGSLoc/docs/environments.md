# Separate environments and repeatable launches

Preserve the dependencies used for accepted experiments, either installed or in a verified backup. AnyGSLoc and the native comparison methods use different Python/PyTorch versions; upstream reconstruction networks also have their own dependencies. The public profile names below select an installed environment without modifying it. An archived environment must be restored before launching its profile.

| Profile | Reference Conda environment | Python | PyTorch / bundled CUDA | Role |
|---|---|---|---|---|
| `main` | `g4splat` | 3.9 | 2.0.1 / 11.8 | AnyGSLoc construction, localization, all downstream ablations, PSNR and figures |
| `native` | `ulfloc_repro` | 3.8 | 2.4.1 / 11.8 | Native STDLoc and ULF-Loc |
| `matcha` | `cybersim_agent` | 3.9 | 2.0.1 / 11.8 | MAtCha native reconstruction |
| `i3dgs` | `i3dgs` | 3.12 | 2.7.1 / 11.8 | i3DGS prior generation/export |
| `long-lrm` | `long_lrm22` | 3.10 | 2.2.0 / 12.1 | Long-LRM prior generation/export |

The names `g4splat` and `cybersim_agent` are historical local environment names. They do not identify the paper's method or a dependency on the unrelated projects sharing those environments. STDLoc and ULF-Loc share the accepted native environment. Their original README installation defaults can differ; this paper's comparisons used CUDA 11.8 for both.

**Evaluating all seven supplied Gaussian priors requires only `main`.** The other three prior-generator environments are needed when regenerating those upstream priors. They are separate from the two environments needed for downstream localization comparisons.

For a smaller installation, keep `main` and `native` available and archive the three upstream environments after preserving their external editable sources. The reference server uses this arrangement after its 2026-10-15 cleanup. All five profiles passed their checks before archiving; `main` and `native` remain installed. The private backup locations and restoration commands are recorded in the server handoff, outside the public source package.

## Check an environment

From the source checkout, using any Python with the standard library:

```bash
python -m reproduction.environment list
python -m reproduction.environment check --profile main
python -m reproduction.environment check --profile native
```

The checker compares the accepted versions, imports critical modules (including the native `simple_knn` extension), and runs a tiny CUDA tensor operation. It records the GPU and available `nvcc` version. Add `--device cpu` for imports/versions only, or `--output runs/environment_checks/native.json` to save a fresh report. A nonzero exit status indicates a mismatch or failed import/runtime check.

Use `--conda /path/to/conda` if Conda is not on PATH, `--env my_native_env` for a renamed environment, or `--prefix /path/to/environment` for an explicit installation. Environment variables and executable paths are resolved at launch; there are no hardcoded server prefixes in these helpers.

Passing a check does not establish that an upstream neural model has all its checkpoints or correct camera preparation. The `main` and `native` profiles additionally have the Shop Facade numerical/measurement validation described in [validation.md](validation.md). Prior-generator profiles have environment checks only; their full training/inference pipelines were not rerun during this release preparation.

## Run without switching the interactive shell

The launcher sets the selected environment's Python, executable/library paths, source checkout, four-thread limits, and known native settings. It propagates `CUDA_VISIBLE_DEVICES` and sets the bundled SuperPoint weight path when present. The interactive shell and Conda environments are not changed.

```bash
python -m reproduction.environment run --profile main -- \
  -m reproduction.run main --config configs/cambridge.json \
  --scene ShopFacade --output runs/shop_rebuild --execute

python -m reproduction.environment run --profile native -- \
  -m reproduction.baselines accuracy --method STDLoc \
  --repo assets/third_party/STDLoc --dataset assets/datasets/Cambridge/ShopFacade \
  --model assets/native/STDLoc/on/ShopFacade --config configs/native/stdloc_cambridge.yaml \
  --mapping-semantics on --scene ShopFacade --output runs/native/STDLoc/on/ShopFacade
```

Replace the Python arguments after `--` with any command in [reproduction.md](reproduction.md). Native timing still uses a separate invocation for every scene/round. The launcher starts one child Python process and does not cache models or query features. For a native prior-generator script, `--cwd /path/to/producer/repository` selects its working directory.

The native compiler defaults are `/usr/bin/gcc`, `/usr/bin/g++`, and architecture `8.6` for the reference RTX 3090. Override `CC`, `CXX`, and `TORCH_CUDA_ARCH_LIST` before launch for another toolchain/GPU. A PyTorch wheel's bundled CUDA runtime and the system CUDA compiler are separate; the checker reports the compiler without claiming that every extension has been rebuilt with it.

On the checked server the system compiler is CUDA 11.8, while Long-LRM's archived PyTorch environment bundles CUDA 12.1. Its installed runtime passed the check before archiving. Rebuilding its CUDA extensions on a new machine additionally requires a toolchain compatible with that PyTorch build; the shared system compiler is not evidence that all five environments can be rebuilt identically.

## Preserve working environments

Capture each environment before changing dependencies:

```bash
python -m reproduction.environment snapshot --profile main \
  --output /backup/anygsloc_envs/main
python -m reproduction.environment snapshot --profile native \
  --output /backup/anygsloc_envs/native
```

Repeat for `matcha`, `i3dgs`, and `long-lrm` if maintaining upstream generation. Each snapshot contains the Conda package URLs/builds, a YAML export, the complete pip inventory, and the selected compilation settings. These are **private restoration records**: they can contain local editable-package paths and are kept outside the public source archive.

For experiments with upgraded dependencies, clone the working environment first:

```bash
conda create --name anygsloc_native_dev --clone ulfloc_repro
python -m reproduction.environment check --profile native --env anygsloc_native_dev
```

Cloning retains local editable-source references. A YAML or `pip freeze` alone does not include those source trees, compiled CUDA extensions, checkpoints, datasets, or the NVIDIA driver. For moving to another server, preserve the corresponding native source snapshots and model assets, recreate the environment following [installation.md](installation.md) / [baselines.md](baselines.md), rebuild extensions for that machine, then rerun the checker and one scene. All five environments were validated in place; a completely fresh recreation of all five environments has not been validated.
