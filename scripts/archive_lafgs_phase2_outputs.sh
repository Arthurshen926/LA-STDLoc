#!/usr/bin/env bash
# Preserve lightweight evidence and the latest state before removing superseded
# early LaFGS/ULF-Loc experiment outputs. The exclusions are current evidence.
set -euo pipefail

POOL_ROOT=${POOL_ROOT:-/mnt/pool/sqy}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-"$POOL_ROOT/stdloc_lafgs_phase2_archive_20260801"}
EXECUTE=${1:-}

if [[ "$EXECUTE" != "--execute" ]]; then
  echo "Usage: $0 --execute" >&2
  exit 2
fi
if [[ ! -d "$POOL_ROOT" || -e "$ARCHIVE_ROOT" ]]; then
  echo "Pool root is missing or archive root already exists: $ARCHIVE_ROOT" >&2
  exit 2
fi

mapfile -d '' CANDIDATES < <(
  find "$POOL_ROOT" -mindepth 1 -maxdepth 1 -type d \
    \( -name 'stdloc_lafgs_r1_*' -o -name 'stdloc_lafgs_mainline*' -o \
       -name 'stdloc_lafgs_cambridge_*' -o -name 'ulfloc_*' \) -print0 | sort -z
)
TARGETS=()
for source in "${CANDIDATES[@]}"; do
  case "$(basename "$source")" in
    stdloc_lafgs_cambridge_matcha2dgs_strict_20260711|stdloc_lafgs_cambridge_best_crossscene_20260711|ulfloc_assets)
      continue
      ;;
  esac
  TARGETS+=("$source")
done

if [[ ${#TARGETS[@]} -ne 85 ]]; then
  echo "Expected exactly 85 audited targets, found ${#TARGETS[@]}. Refusing to continue." >&2
  exit 3
fi

mkdir -p "$ARCHIVE_ROOT"/{metadata,final_state}
printf 'source\tbytes\n' > "$ARCHIVE_ROOT/target_manifest.tsv"
printf 'source\tselected_file\tmtime_epoch\tbytes\n' > "$ARCHIVE_ROOT/final_state_manifest.tsv"

archive_metadata() {
  local source=$1 destination=$2 path relative
  while IFS= read -r -d '' path; do
    relative=${path#"$source/"}
    mkdir -p "$destination/$(dirname "$relative")"
    cp --preserve=mode,timestamps "$path" "$destination/$relative"
  done < <(
    find "$source" -type f \
      \( -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o \
         -name '*.md' -o -name '*.txt' -o -name '*.log' -o -name '*.csv' \) \
      -print0
  )
  return 0
}

archive_latest_state() {
  local source=$1 root_name=$2 pattern=$3 target_name=$4 record path mtime bytes
  record=$(find "$source" -type f -name "$pattern" -printf '%T@\t%s\t%p\n' | sort -nr | head -n 1 || true)
  [[ -n "$record" ]] || return 0
  IFS=$'\t' read -r mtime bytes path <<< "$record"
  mkdir -p "$ARCHIVE_ROOT/final_state/$root_name"
  cp --reflink=auto --preserve=mode,timestamps "$path" "$ARCHIVE_ROOT/final_state/$root_name/$target_name"
  printf '%s\t%s\t%s\t%s\n' "$source" "$path" "$mtime" "$bytes" >> "$ARCHIVE_ROOT/final_state_manifest.tsv"
}

for source in "${TARGETS[@]}"; do
  root_name=$(basename "$source")
  printf '%s\t%s\n' "$source" "$(du -sb "$source" | awk '{print $1}')" >> "$ARCHIVE_ROOT/target_manifest.tsv"
  archive_metadata "$source" "$ARCHIVE_ROOT/metadata/$root_name"
  archive_latest_state "$source" "$root_name" 'point_cloud.ply' 'point_cloud.ply'
  archive_latest_state "$source" "$root_name" 'loc_state.pt' 'loc_state.pt'
done

find "$ARCHIVE_ROOT/metadata" -type f -printf '%p\n' | sort > "$ARCHIVE_ROOT/metadata_files.txt"
find "$ARCHIVE_ROOT/final_state" -type f -printf '%p\n' | sort > "$ARCHIVE_ROOT/final_state_files.txt"

for source in "${TARGETS[@]}"; do
  case "$source" in
    "$POOL_ROOT"/stdloc_lafgs_r1_*|"$POOL_ROOT"/stdloc_lafgs_mainline*|"$POOL_ROOT"/stdloc_lafgs_cambridge_*|"$POOL_ROOT"/ulfloc_*)
      rm -rf --one-file-system -- "$source"
      ;;
    *)
      echo "Refusing to delete unexpected path: $source" >&2
      exit 4
      ;;
  esac
done

sync
echo "Archived and removed ${#TARGETS[@]} phase-2 legacy roots."
du -sh "$ARCHIVE_ROOT"
