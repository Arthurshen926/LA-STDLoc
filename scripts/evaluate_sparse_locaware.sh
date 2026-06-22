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

"$PYTHON" stdloc.py \
  -s "$DATA_ROOT/$SCENE" \
  -m "$MODEL_ROOT/$SCENE" \
  -r 1 \
  -f sp \
  -g 3dgs \
  --images processed \
  --iteration "${ITERATION:-3000}" \
  --cfg "${CFG:-configs/stdloc_cambridge.yaml}" \
  --prefix "la-sparse" \
  --sparse_only
