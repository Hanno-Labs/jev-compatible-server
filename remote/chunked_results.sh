#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 restore|seal|loop|publish REMOTE_DIR LOCAL_DIR" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
command_name="$1"
remote_dir="${2%/}"
local_dir="${3%/}"
chunk_rows="${RESULTS_CHUNK_ROWS:-1000}"
sync_interval="${RESULTS_CHUNK_INTERVAL_SECONDS:-60}"
chunk_dir="$local_dir/chunks"
raw_path="$local_dir/raw.jsonl"

[[ "$chunk_rows" =~ ^[1-9][0-9]*$ ]] || {
  echo "RESULTS_CHUNK_ROWS must be a positive integer" >&2
  exit 2
}
[[ "$sync_interval" =~ ^[1-9][0-9]*$ ]] || {
  echo "RESULTS_CHUNK_INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
}

mkdir -p "$local_dir" "$chunk_dir"

sorted_chunks() {
  find "$chunk_dir" -maxdepth 1 -type f -name 'rows-*.jsonl' -print0 | sort -z
}

restore_chunks() {
  rm -f "$chunk_dir"/*.uploaded
  hf sync "$remote_dir/chunks" "$chunk_dir" >/dev/null 2>&1 || true

  local found=0
  local expected_start=1
  local chunk base start end count
  while IFS= read -r -d '' chunk; do
    found=1
    base="${chunk##*/}"
    if [[ ! "$base" =~ ^rows-([0-9]{8})-([0-9]{8})\.jsonl$ ]]; then
      echo "invalid durable chunk name: $base" >&2
      exit 1
    fi
    start=$((10#${BASH_REMATCH[1]}))
    end=$((10#${BASH_REMATCH[2]}))
    if (( start != expected_start || end < start )); then
      echo "non-contiguous durable chunk: $base expected_start=$expected_start" >&2
      exit 1
    fi
    count=$(wc -l < "$chunk")
    if (( count != end - start + 1 )); then
      echo "durable chunk line-count mismatch: $base count=$count" >&2
      exit 1
    fi
    expected_start=$((end + 1))
    touch "$chunk.uploaded"
  done < <(sorted_chunks)

  if (( found )); then
    : > "$raw_path"
    while IFS= read -r -d '' chunk; do
      cat "$chunk" >> "$raw_path"
    done < <(sorted_chunks)
    echo "DECISION_BENCH_CHUNK_RESTORE records=$((expected_start - 1)) source=chunks" >&2
    hf buckets cp "$remote_dir/summary.json" "$local_dir/summary.json" >/dev/null 2>&1 || true
    hf buckets cp "$remote_dir/manifest.json" "$local_dir/manifest.json" >/dev/null 2>&1 || true
    return
  fi

  if hf buckets cp "$remote_dir/raw.jsonl" "$raw_path" >/dev/null 2>&1; then
    echo "DECISION_BENCH_CHUNK_RESTORE records=$(wc -l < "$raw_path") source=legacy_raw" >&2
  else
    : > "$raw_path"
    echo "DECISION_BENCH_CHUNK_RESTORE records=0 source=empty" >&2
  fi
  hf buckets cp "$remote_dir/summary.json" "$local_dir/summary.json" >/dev/null 2>&1 || true
  hf buckets cp "$remote_dir/manifest.json" "$local_dir/manifest.json" >/dev/null 2>&1 || true
}

uploaded_record_count() {
  local last_end=0
  local marker base end
  for marker in "$chunk_dir"/rows-*.jsonl.uploaded; do
    [[ -e "$marker" ]] || continue
    base="${marker##*/}"
    if [[ "$base" =~ ^rows-[0-9]{8}-([0-9]{8})\.jsonl\.uploaded$ ]]; then
      end=$((10#${BASH_REMATCH[1]}))
      (( end > last_end )) && last_end=$end
    fi
  done
  printf '%s\n' "$last_end"
}

seal_chunks() {
  local force="$1"
  [[ -f "$raw_path" ]] || return 0

  local total limit start end expected chunk_name chunk_path temporary actual
  total=$(wc -l < "$raw_path")
  start=$(($(uploaded_record_count) + 1))
  if [[ "$force" == "true" ]]; then
    limit=$total
  else
    limit=$((total - (total % chunk_rows)))
  fi

  while (( start <= limit )); do
    end=$((start + chunk_rows - 1))
    (( end > limit )) && end=$limit
    printf -v chunk_name 'rows-%08d-%08d.jsonl' "$start" "$end"
    chunk_path="$chunk_dir/$chunk_name"
    temporary=$(mktemp "$chunk_dir/.${chunk_name}.XXXXXX")
    sed -n "${start},${end}p" "$raw_path" > "$temporary"
    actual=$(wc -l < "$temporary")
    expected=$((end - start + 1))
    if (( actual != expected )); then
      rm -f "$temporary"
      echo "refusing incomplete result chunk: $chunk_name expected=$expected actual=$actual" >&2
      exit 1
    fi
    mv "$temporary" "$chunk_path"
    hf buckets cp "$chunk_path" "$remote_dir/chunks/$chunk_name" >/dev/null
    touch "$chunk_path.uploaded"
    echo "DECISION_BENCH_CHUNK_UPLOADED start=$start end=$end records=$expected" >&2
    start=$((end + 1))
  done
}

publish_final() {
  seal_chunks true
  for name in raw.jsonl summary.json manifest.json; do
    [[ -f "$local_dir/$name" ]] || {
      echo "missing final result artifact: $local_dir/$name" >&2
      exit 1
    }
    hf buckets cp "$local_dir/$name" "$remote_dir/$name" >/dev/null
  done
  echo "DECISION_BENCH_FINAL_PUBLISHED remote=$remote_dir records=$(wc -l < "$raw_path")" >&2
}

case "$command_name" in
  restore)
    restore_chunks
    ;;
  seal)
    seal_chunks true
    ;;
  loop)
    while true; do
      sleep "$sync_interval"
      seal_chunks false
    done
    ;;
  publish)
    publish_final
    ;;
  *)
    usage
    ;;
esac
