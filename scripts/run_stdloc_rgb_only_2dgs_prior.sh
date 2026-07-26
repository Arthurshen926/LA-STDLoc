#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 <gpu>" >&2
  exit 2
fi

GPU="$1"
case "$GPU" in 0|1|2) ;; *) echo "GPU must be 0, 1, or 2" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
EXPERIMENT_ROOT="${LAFGS_SANITIZATION_ROOT:-/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725}"
SCENE="OldHospital"
TRAIN_ROOT="$EXPERIMENT_ROOT/$SCENE/rgb_only_2dgs_stdloc_train"
PRIOR_ROOT="$EXPERIMENT_ROOT/$SCENE/rgb_only_2dgs_stdloc"
POINT_CLOUD="$TRAIN_ROOT/point_cloud/iteration_30000/point_cloud.ply"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0

cd "$REPO_ROOT"
if [[ ! -f "$POINT_CLOUD" ]]; then
  "$PYTHON" train.py \
    --source_path "$DATA_ROOT/$SCENE" \
    --model_path "$TRAIN_ROOT" \
    --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp --resolution 1 \
    --longest_edge 640 --iterations 30000 \
    --rgb_only_reconstruction \
    --densify_grad_threshold 0.0004 \
    --position_lr_init 0.000016 --scaling_lr 0.001 \
    --test_iterations 30000 --save_iterations 30000
fi

[[ -f "$TRAIN_ROOT/rgb_reconstruction_manifest.json" ]] || {
  echo "Missing RGB-only training manifest: $TRAIN_ROOT" >&2
  exit 1
}

if [[ ! -f "$PRIOR_ROOT/rgb_prior_manifest.json" ]]; then
  "$PYTHON" scripts/export_rgb_gaussian_prior.py \
    --input_ply "$POINT_CLOUD" \
    --output_model "$PRIOR_ROOT" \
    --gaussian_type 2dgs --sh_degree 3 \
    --source_path "$DATA_ROOT/$SCENE" --images processed \
    --longest_edge 0 --iteration 30000 --prior_kind rgb_only
fi

echo "$PRIOR_ROOT"
