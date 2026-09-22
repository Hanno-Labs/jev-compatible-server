#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ARCHIVE:?SOURCE_ARCHIVE is required}"
: "${RESULTS_BUCKET:?RESULTS_BUCKET is required}"
: "${DECISION_REGISTRY:?DECISION_REGISTRY is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"

source_root=/workflow/decision-bench
output_root=/workflow/results
server_root=/workflow/jev-compatible-server
result_dir="$output_root/$MODEL_KEY"
mkdir -p "$source_root" "$output_root" "$result_dir"

if [[ ! -f /workflow/decision-bench-runtime.tgz ]]; then
  hf buckets cp "$SOURCE_ARCHIVE" /workflow/decision-bench-runtime.tgz
fi
tar -xzf /workflow/decision-bench-runtime.tgz -C "$source_root" --strip-components=1
hf sync "$RESULTS_BUCKET" "$output_root" || true

uv sync --directory "$source_root" --extra hf
uv pip install --python "$source_root/.venv/bin/python" "$server_root[transformers]"

sync_results() {
  while [[ -f /workflow/sync.running ]]; do
    sleep "${RESULTS_SYNC_INTERVAL_SECONDS:-60}"
    hf sync "$output_root" "$RESULTS_BUCKET"
  done
  hf sync "$output_root" "$RESULTS_BUCKET"
}

cleanup() {
  rm -f /workflow/sync.running
  if [[ -n "${sync_pid:-}" ]]; then
    kill "$sync_pid" 2>/dev/null || true
    wait "$sync_pid" 2>/dev/null || true
  fi
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT

if [[ -f "$result_dir/summary.json" ]] && jq -e '.requested_rows == 23900 and (.successful_rows + .error_rows == 23900)' "$result_dir/summary.json" >/dev/null; then
  echo "DECISION_BENCH_SKIP model=$MODEL_KEY reason=complete" >&2
  exit 0
fi

touch /workflow/sync.running
sync_results &
sync_pid=$!

DECISION_REGISTRY="$DECISION_REGISTRY" \
  DECISION_MAX_BATCH_SIZE="${DECISION_MAX_BATCH_SIZE:-8}" \
  DECISION_BATCH_WAIT_MS=5 \
  "$source_root/.venv/bin/jev-compatible-server" \
  --model-batch-size "${MODEL_BATCH_SIZE:-8}" >/workflow/server.log 2>&1 &
server_pid=$!

attempts=0
until curl -fsS http://127.0.0.1:8000/health >/dev/null; do
  attempts=$((attempts + 1))
  if (( attempts >= 720 )); then
    sed -n '1,240p' /workflow/server.log >&2 || true
    exit 1
  fi
  sleep 5
done

(cd "$source_root" && uv run python -m decision_bench.cli run-jev-server \
  "$source_root/task_specs/decisionbench-dev.toml" "$result_dir" \
  --project-root "$source_root" --base-url http://127.0.0.1:8000 \
  --model "$MODEL_KEY" --concurrency "${EVAL_CONCURRENCY:-8}")

jq -e '.requested_rows == 23900 and (.successful_rows + .error_rows == 23900)' "$result_dir/summary.json" >/dev/null
hf sync "$result_dir" "$RESULTS_BUCKET/$MODEL_KEY"
echo "DECISION_BENCH_COMPLETE model=$MODEL_KEY" >&2
