#!/usr/bin/env bash
# Archive reproducibility metadata and one final state per legacy experiment,
# then remove only explicitly scoped obsolete output roots.
set -euo pipefail

POOL_ROOT=${POOL_ROOT:-/mnt/pool/sqy}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-"$POOL_ROOT/stdloc_lafgs_legacy_archive_20260801"}
EXECUTE=${1:-}

if [[ "$EXECUTE" != "--execute" ]]; then
  echo "Usage: $0 --execute" >&2
  exit 2
fi

if [[ ! -d "$POOL_ROOT" || -e "$ARCHIVE_ROOT" ]]; then
  echo "Pool root is missing or archive root already exists: $ARCHIVE_ROOT" >&2
  exit 2
fi

mapfile -d '' TARGETS < <(
  find "$POOL_ROOT" -mindepth 1 -maxdepth 1 -type d \
    \( -name 'stdloc_la_*' -o -name 'stdloc_lafgs_v2*' \) -print0 | sort -z
)
VIEWPOINT_TARGETS=(
  "$POOL_ROOT/stdloc_lafgs_viewpoint_candidate_20260801"
  "$POOL_ROOT/stdloc_lafgs_viewpoint_completion_v2_20260801"
)

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "No legacy targets found beneath $POOL_ROOT" >&2
  exit 2
fi

mkdir -p "$ARCHIVE_ROOT"/{metadata,final_state,viewpoint}
printf 'source\tbytes\n' > "$ARCHIVE_ROOT/target_manifest.tsv"
printf 'source\tselected_file\tmtime_epoch\tbytes\n' > "$ARCHIVE_ROOT/final_state_manifest.tsv"

archive_metadata() {
  local source=$1 destination=$2
  local path relative
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
  local source=$1 root_name=$2 pattern=$3 target_name=$4
  local record path mtime bytes
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

for source in "${VIEWPOINT_TARGETS[@]}"; do
  [[ -d "$source" ]] || continue
  root_name=$(basename "$source")
  printf '%s\t%s\n' "$source" "$(du -sb "$source" | awk '{print $1}')" >> "$ARCHIVE_ROOT/target_manifest.tsv"
  archive_metadata "$source" "$ARCHIVE_ROOT/viewpoint/$root_name"
done

find "$ARCHIVE_ROOT/metadata" -type f -printf '%p\n' | sort > "$ARCHIVE_ROOT/metadata_files.txt"
find "$ARCHIVE_ROOT/viewpoint" -type f -printf '%p\n' | sort > "$ARCHIVE_ROOT/viewpoint_files.txt"
find "$ARCHIVE_ROOT/final_state" -type f -printf '%p\n' | sort > "$ARCHIVE_ROOT/final_state_files.txt"

for source in "${TARGETS[@]}" "${VIEWPOINT_TARGETS[@]}"; do
  [[ -d "$source" ]] || continue
  case "$source" in
    "$POOL_ROOT"/stdloc_la_*|"$POOL_ROOT"/stdloc_lafgs_v2*|"$POOL_ROOT"/stdloc_lafgs_viewpoint_candidate_20260801|"$POOL_ROOT"/stdloc_lafgs_viewpoint_completion_v2_20260801)
      rm -rf --one-file-system -- "$source"
      ;;
    *)
      echo "Refusing to delete unexpected path: $source" >&2
      exit 3
      ;;
  esac
done

sync
echo "Archived and removed ${#TARGETS[@]} legacy roots plus viewpoint no-go roots."
du -sh "$ARCHIVE_ROOT"
