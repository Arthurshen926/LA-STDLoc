#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
SCENE=${SCENE:-ShopFacade}
MODEL_ROOT=${MODEL_ROOT:-map_cambridge_la}
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/miniconda3/envs/ulfloc_repro/bin/python ]]; then
    PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python
  else
    PYTHON=python
  fi
fi

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8}
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

"$PYTHON" train_locaware.py \
  -s "$DATA_ROOT/$SCENE" \
  -m "$MODEL_ROOT/$SCENE" \
  -r 1 \
  -f sp \
  -g 3dgs \
  --iterations 3000 \
  --data_device cpu \
  --images processed \
  --densify_grad_threshold 0.0004 \
  --position_lr_init 0.000016 \
  --scaling_lr 0.001 \
  --loc_interval 8 \
  --loc_anchors 1024 \
  --loc_desc_weight 1.0 \
  --loc_reproj_weight 0.1 \
  --loc_proto_weight 0.1 \
  --loc_opacity_weight 0.001 \
  --save_iterations 3000 \
  --test_iterations 3000
