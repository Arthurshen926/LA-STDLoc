# Installation

For existing working environments, use the [environment launcher and checks](environments.md) to select `main` or `native` without manually switching shells. The instructions below describe installing the main environment.

Run commands from the release repository root. The numerical reference environment is Linux, Python 3.9, PyTorch **2.0.1+cu118**, torchvision **0.15.2+cu118**, NumPy **1.26.4**, PoseLib **2.0.5**, and gsplat **1.5.3**. Formal timing used an RTX 3090 with four CPU threads. Other environments may run, but are not the reference for identical floating-point results or latency.

```bash
conda create -n anygsloc python=3.9 -y
conda activate anygsloc
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e '.[mapping,test]'
python -m pip install numpy==1.26.4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MAX_JOBS=4
```

Rendering requires a CUDA compiler compatible with CUDA 11.8 and a working C++ toolchain. Activate the environment so that its `ninja` executable is on `PATH`; invoking an absolute Python executable alone does not expose its sibling tools. The first render may compile gsplat kernels. `pip install -e .` installs only the localization dependencies; add `[mapping]` for offline conversion/PSNR.

Obtain the original SuperPoint weights specified by `anygsloc/features/superpoint.py`, then set:

```bash
export ANYGSLOC_SUPERPOINT_WEIGHTS="$PWD/assets/weights/superpoint_v1.pth"
sha256sum "$ANYGSLOC_SUPERPOINT_WEIGHTS"
# 52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e
python -m anygsloc --help
python -m pytest -q
```

The loader rejects different weights. It does not download files implicitly. Native STDLoc/ULF-Loc require their own CUDA extensions and use the separate environment documented in [baselines.md](baselines.md).

The CPU-only reference aggregation requires just NumPy and the source checkout; it does not import PyTorch, download weights, or load datasets.

If the system C++ runtime is older than the one used to build gsplat, activate the environment and expose its runtime: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`. The reference environment used Pillow 9.5.0 and OpenCV-headless 4.12.0.88; their installed versions are recorded in `reproduction/environment_g4splat.json`. OpenCV 4.12 declares a NumPy 2 requirement although this accepted environment ran it with NumPy 1.26.4; an exact environment restoration therefore requires installing that OpenCV wheel with `--no-deps` after the pinned NumPy. A fresh dependency-resolved installation may choose an earlier compatible OpenCV wheel. Record this difference when comparing numerical results.
