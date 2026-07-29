#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/lafgs_poselib}"
EIGEN3_DIR="${EIGEN3_DIR:-/root/miniconda3/envs/g4splat/share/eigen3/cmake}"
BUILD_JOBS="${BUILD_JOBS:-16}"
POSELIB_SOURCE_DIR="${POSELIB_SOURCE_DIR:-$ROOT/third_party/PoseLib}"
POSELIB_URL="${POSELIB_URL:-https://github.com/PoseLib/PoseLib.git}"
POSELIB_COMMIT="${POSELIB_COMMIT:-7e9f5f53372e43f89655040d4dfc4a00e5ace11c}"

if [[ ! -d "$POSELIB_SOURCE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$POSELIB_SOURCE_DIR")"
  git clone --filter=blob:none "$POSELIB_URL" "$POSELIB_SOURCE_DIR"
fi

if ! git -C "$POSELIB_SOURCE_DIR" cat-file -e "${POSELIB_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$POSELIB_SOURCE_DIR" fetch --depth=1 origin "$POSELIB_COMMIT"
fi
git -C "$POSELIB_SOURCE_DIR" checkout --detach "$POSELIB_COMMIT"
git -C "$POSELIB_SOURCE_DIR" submodule update --init --recursive

actual_commit="$(git -C "$POSELIB_SOURCE_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$POSELIB_COMMIT" ]]; then
  echo "PoseLib commit mismatch: expected $POSELIB_COMMIT, got $actual_commit" >&2
  exit 1
fi

cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
  -DPython_EXECUTABLE="$PYTHON" \
  -DPython_ROOT_DIR="$(dirname "$(dirname "$PYTHON")")" \
  -DEigen3_DIR="$EIGEN3_DIR" \
  -DPOSELIB_SOURCE_DIR="$POSELIB_SOURCE_DIR" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target _lafgs_poselib -j"$BUILD_JOBS"

"$PYTHON" - <<'PY'
from localization_training.dependency_pose_sampler import compiled_backend_available
if not compiled_backend_available():
    raise SystemExit("compiled LaFGS PoseLib backend was not importable")
print("compiled LaFGS PoseLib backend: ready")
PY
