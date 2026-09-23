#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_ARCHIVE:?SOURCE_ARCHIVE is required}"
: "${RESULTS_BUCKET:?RESULTS_BUCKET is required}"
: "${DECISION_REGISTRY:?DECISION_REGISTRY is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"
run_mode="${RUN_MODE:-suite}"
[[ "$run_mode" == suite || "$run_mode" == probe ]] || {
  echo "RUN_MODE must be suite or probe" >&2
  exit 2
}

source_root=/workflow/decision-bench
output_root=/workflow/results
server_root=/workflow/jev-compatible-server
result_dir="$output_root/$MODEL_KEY"
chunk_helper=/workflow/chunked_results.sh
remote_result_dir="$RESULTS_BUCKET/$MODEL_KEY"
mkdir -p "$source_root" "$output_root" "$result_dir"

if [[ ! -f /workflow/decision-bench-runtime.tgz ]]; then
  hf buckets cp "$SOURCE_ARCHIVE" /workflow/decision-bench-runtime.tgz
fi
tar -xzf /workflow/decision-bench-runtime.tgz -C "$source_root" --strip-components=1
if [[ "$run_mode" == suite ]]; then
  bash "$chunk_helper" restore "$remote_result_dir" "$result_dir"
fi

uv sync --directory "$source_root" --extra hf
uv pip install --python "$source_root/.venv/bin/python" "$server_root[transformers]"
if [[ "${JEV_LOCAL_OPTIMIZED:-0}" == "1" ]]; then
  uv pip install --python "$source_root/.venv/bin/python" \
    'flash-linear-attention[cuda]==0.5.2' \
    'causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
  echo "DECISION_BENCH_JEV_LOCAL_KERNEL optimized=1" >&2
fi
if [[ "${DECISION_NATIVE_EOS:-0}" == "1" ]]; then
  uv pip install --python "$source_root/.venv/bin/python" \
    'torch==2.9.1' 'transformers==5.17.0' 'flash-linear-attention==0.5.2'
fi

cleanup() {
  if [[ -n "${chunk_pid:-}" ]]; then
    kill "$chunk_pid" 2>/dev/null || true
    wait "$chunk_pid" 2>/dev/null || true
    chunk_pid=
  fi
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ "$run_mode" == suite ]]; then
    bash "$chunk_helper" seal "$remote_result_dir" "$result_dir" || true
  fi
}
trap cleanup EXIT TERM INT

wait_for_server() {
  local attempts=0
  local server_status=0
  until curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      if wait "$server_pid"; then
        server_status=0
      else
        server_status=$?
      fi
      echo "DECISION_BENCH_SERVER_EXITED status=$server_status" >&2
      sed -n '1,240p' /workflow/server.log >&2 || true
      return 1
    fi
    attempts=$((attempts + 1))
    if (( attempts >= 720 )); then
      echo "DECISION_BENCH_SERVER_TIMEOUT" >&2
      sed -n '1,240p' /workflow/server.log >&2 || true
      return 1
    fi
    sleep 5
  done
}

summary_is_complete() {
  if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then
    jq -e '.requested_rows == 23900 and .successful_rows == 23900 and .error_rows == 0' "$1" >/dev/null
  else
    jq -e '.requested_rows == 23900 and (.successful_rows + .error_rows == 23900)' "$1" >/dev/null
  fi
}

if [[ "$run_mode" == suite && -f "$result_dir/summary.json" ]] && summary_is_complete "$result_dir/summary.json"; then
  echo "DECISION_BENCH_SKIP model=$MODEL_KEY reason=complete" >&2
  exit 0
fi

if [[ "$run_mode" == suite ]]; then
  bash "$chunk_helper" loop "$remote_result_dir" "$result_dir" &
  chunk_pid=$!
fi

DECISION_REGISTRY="$DECISION_REGISTRY" \
  DECISION_MAX_BATCH_SIZE="${DECISION_MAX_BATCH_SIZE:-8}" \
  DECISION_BATCH_WAIT_MS=5 \
  "$source_root/.venv/bin/jev-compatible-server" \
  --model-batch-size "${MODEL_BATCH_SIZE:-8}" >/workflow/server.log 2>&1 &
server_pid=$!
wait_for_server

if [[ "$run_mode" == probe ]]; then
  curl -fsS http://127.0.0.1:8000/v1/systemone \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_KEY\",\"state\":\"The light is on.\",\"questions\":{\"decision\":{\"type\":\"noul\",\"instructions\":\"Is the light on?\"}}}" \
    >/workflow/probe-response.json
  jq -e '.answers.decision.type == "noul" and (.answers.decision.noul | type == "number")' /workflow/probe-response.json >/dev/null
  if [[ "$MODEL_KEY" == system-one-sg ]]; then
    jq -nc '{model:"system-one-sg",state:"The light is on.",questions:{many:{type:"choice",instructions:"Choose the matching option.",criteria:(reduce range(11) as $i ({}; . + {("c"+($i|tostring)):("candidate "+($i|tostring))}))}}}' |
      curl -fsS http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d @- >/workflow/probe-eleven-response.json
    jq -e '.answers.many.type == "choice" and (.answers.many.probabilities | length == 11)' /workflow/probe-eleven-response.json >/dev/null
  fi
  echo "DECISION_BENCH_PROBE_COMPLETE model=$MODEL_KEY" >&2
  exit 0
fi

if [[ "${DECISION_NATIVE_EOS:-0}" == "1" ]]; then
  curl -fsS --max-time 180 \
    -H 'Content-Type: application/json' \
    -d '{"model":"decision-1.0-eos-0.8b","state":"A customer needs help with a billing issue.","questions":{"route":{"type":"choice","instructions":"Choose the support route.","criteria":{"self":"Self service","human":"Human support"}}}}' \
    http://127.0.0.1:8000/v1/systemone \
    | jq -e '.answers.route.type == "choice" and (.answers.route.probabilities | keys == ["human", "self"])' >/dev/null
  echo 'DECISION_BENCH_NATIVE_PROBE_COMPLETE model=decision-1.0-eos-0.8b' >&2
fi

(cd "$source_root" && "$source_root/.venv/bin/python" -m decision_bench.cli run-jev-server \
  "$source_root/task_specs/decisionbench-dev.toml" "$result_dir" \
  --project-root "$source_root" --base-url http://127.0.0.1:8000 \
  --model "$MODEL_KEY" --concurrency "${EVAL_CONCURRENCY:-8}")

summary_is_complete "$result_dir/summary.json"
kill "$chunk_pid" 2>/dev/null || true
wait "$chunk_pid" 2>/dev/null || true
chunk_pid=
bash "$chunk_helper" publish "$remote_result_dir" "$result_dir"
echo "DECISION_BENCH_COMPLETE model=$MODEL_KEY" >&2
