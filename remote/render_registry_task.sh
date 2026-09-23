#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 && $# -ne 6 ]]; then
  echo "usage: $0 TASK_NAME RESULT_NAME MODEL_KEY BATCH_SIZE GPU_MEMORY_GB [suite|probe]" >&2
  exit 2
fi

task_name="$1"
result_name="$2"
model_key="$3"
batch_size="$4"
gpu_memory="$5"
run_mode="${6:-suite}"
retry_errors="${RETRY_ERRORS:-0}"
results_chunk_rows="${RESULTS_CHUNK_ROWS:-1000}"
template="$(cd "$(dirname "$0")" && pwd)/decisionbench_registry_base.dstack.yml"

for value in "$task_name" "$result_name" "$model_key"; do
  [[ "$value" =~ ^[A-Za-z0-9._/-]+$ ]] || {
    echo "task, result, and model identifiers must use safe path characters" >&2
    exit 2
  }
done
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "batch size must be a positive integer" >&2
  exit 2
}
[[ "$gpu_memory" =~ ^[1-9][0-9]*$ ]] || {
  echo "GPU memory must be a positive integer number of GB" >&2
  exit 2
}

[[ "$run_mode" == suite || "$run_mode" == probe ]] || {
  echo "run mode must be suite or probe" >&2
  exit 2
}
[[ "$retry_errors" == 0 || "$retry_errors" == 1 ]] || {
  echo "RETRY_ERRORS must be 0 or 1" >&2
  exit 2
}
[[ "$results_chunk_rows" =~ ^[1-9][0-9]*$ ]] || {
  echo "RESULTS_CHUNK_ROWS must be a positive integer" >&2
  exit 2
}

sed \
  -e "s/__TASK_NAME__/$task_name/g" \
  -e "s/__RUN_NAME__/$result_name/g" \
  -e "s/__MODEL_KEY__/$model_key/g" \
  -e "s/__BATCH_SIZE__/$batch_size/g" \
  -e "s/__GPU_MEMORY__/$gpu_memory/g" \
  -e "s/__RUN_MODE__/$run_mode/g" \
  -e "s/__RETRY_ERRORS__/$retry_errors/g" \
  -e "s/__RESULTS_CHUNK_ROWS__/$results_chunk_rows/g" \
  "$template"
