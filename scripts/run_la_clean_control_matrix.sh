#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_DAY=${RUN_DAY:-$(date +%Y%m%d)}
LOG_ROOT=${LOG_ROOT:-/mnt/pool/sqy/stdloc_la_clean_mainline_logs_${RUN_STAMP}}
OUT_ROOT_8192=${OUT_ROOT_8192:-/mnt/pool/sqy/stdloc_la_clean_mainline_control_8192_2000_${RUN_DAY}}
OUT_ROOT_16384=${OUT_ROOT_16384:-/mnt/pool/sqy/stdloc_la_clean_mainline_control_16384_2000_${RUN_DAY}}
LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-2000}

RUN_SHOP_8192=${RUN_SHOP_8192:-1}
RUN_OLD_8192=${RUN_OLD_8192:-1}
RUN_OLD_16384=${RUN_OLD_16384:-1}

GPU_SHOP_8192=${GPU_SHOP_8192:-0}
GPU_OLD_8192=${GPU_OLD_8192:-1}
GPU_OLD_16384=${GPU_OLD_16384:-2}

SEED_SHOP_8192=${SEED_SHOP_8192:-301}
SEED_OLD_8192=${SEED_OLD_8192:-302}
SEED_OLD_16384=${SEED_OLD_16384:-303}

mkdir -p "$LOG_ROOT"

run_one() {
  local scene=$1
  local capacity=$2
  local seed=$3
  local gpu=$4
  local out_root=$5
  local log_name=$6
  shift 6
  local extra_env=("$@")

  (
    set -o pipefail
    export SCENES="$scene"
    export LA_ADAPT_STEPS="$LA_ADAPT_STEPS"
    export TRAIN_SEED="$seed"
    export GPU="$gpu"
    export OUT_ROOT="$out_root"
    export LA_BOOTSTRAP_LANDMARK_NUM="$capacity"
    export LA_DETECTOR_LANDMARK_NUM="$capacity"
    for item in "${extra_env[@]}"; do
      export "$item"
    done
    bash "$SCRIPT_DIR/run_la_clean_real_train_mainline.sh" 2>&1 | tee "$LOG_ROOT/$log_name"
  ) &
}

if [[ "$RUN_SHOP_8192" == "1" ]]; then
  run_one ShopFacade 8192 "$SEED_SHOP_8192" "$GPU_SHOP_8192" "$OUT_ROOT_8192" shop8192.log
fi

if [[ "$RUN_OLD_8192" == "1" ]]; then
  run_one OldHospital 8192 "$SEED_OLD_8192" "$GPU_OLD_8192" "$OUT_ROOT_8192" old8192.log
fi

if [[ "$RUN_OLD_16384" == "1" ]]; then
  run_one OldHospital 16384 "$SEED_OLD_16384" "$GPU_OLD_16384" "$OUT_ROOT_16384" old16384.log
fi

wait
