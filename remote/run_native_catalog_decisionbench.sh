#!/usr/bin/env bash
# Run one public native DecisionBench catalog model behind jev-compatible-server.
#
# Model-specific code stays here rather than in dstack YAML so a retry resumes the
# same immutable result chunks and the scheduler remains only a portable harness.
set -euo pipefail

: "${SOURCE_ARCHIVE:?SOURCE_ARCHIVE is required}"
: "${RESULTS_BUCKET:?RESULTS_BUCKET is required}"
: "${DECISION_REGISTRY:?DECISION_REGISTRY is required}"
: "${MODEL_KEY:?MODEL_KEY is required}"

source_root=/workflow/decision-bench
server_root=/workflow/jev-compatible-server
native_root=/workflow/native-source
output_root=/workflow/results
result_dir="$output_root/$MODEL_KEY"
remote_result_dir="$RESULTS_BUCKET/$MODEL_KEY"
chunk_helper=/workflow/chunked_results.sh
compat_port=8100
native_pid=
compat_pid=
chunk_pid=
active_result_dir=
active_remote_dir=

mkdir -p "$source_root" "$server_root" "$native_root" "$output_root" /workflow/models

start_process() {
  local name="$1"
  shift
  "$@" >"/workflow/${name}.log" 2>&1 &
  started_pid=$!
}

stop_process() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

show_log() {
  local name="$1"
  echo "NATIVE_CATALOG_${name}_LOG" >&2
  awk '{ lines[NR % 500] = $0 } END { start = NR > 500 ? NR - 499 : 1; for (i = start; i <= NR; i++) print lines[i % 500] }' "/workflow/${name}.log" >&2 || true
}

cleanup() {
  stop_process "$chunk_pid"
  stop_process "$compat_pid"
  stop_process "$native_pid"
  if [[ -n "$active_result_dir" && -n "$active_remote_dir" ]]; then
    bash "$chunk_helper" seal "$active_remote_dir" "$active_result_dir" || true
  fi
}
trap cleanup EXIT TERM INT

wait_for_http() {
  local name="$1"
  local pid="$2"
  local url="$3"
  local attempts=0
  until curl -fsS "$url" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "NATIVE_CATALOG_${name}_EXITED model=$MODEL_KEY" >&2
      show_log "$name"
      return 1
    fi
    attempts=$((attempts + 1))
    if (( attempts >= 720 )); then
      echo "NATIVE_CATALOG_${name}_TIMEOUT model=$MODEL_KEY url=$url" >&2
      show_log "$name"
      return 1
    fi
    sleep 5
  done
}

clone_pinned() {
  local repo="$1"
  local revision="$2"
  local destination="$3"
  git clone --filter=blob:none "$repo" "$destination"
  git -C "$destination" checkout --detach "$revision"
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]]
}

install_compat_server() {
  uv venv /workflow/compat-venv --python 3.11
  uv pip install --python /workflow/compat-venv/bin/python "$server_root"
}

start_compat_server() {
  start_process compat /workflow/compat-venv/bin/python -m uvicorn jev_compatible_server.app:create_app --factory --host 127.0.0.1 --port "$compat_port"
  compat_pid=$started_pid
  wait_for_http compat "$compat_pid" "http://127.0.0.1:${compat_port}/health"
}

start_native_server() {
  local native_python
  case "$MODEL_KEY" in
    djev)
      # djev-dev's runtime patch is source-pinned and applies to the pinned vLLM
      # image selected by the renderer for this model.
      clone_pinned https://github.com/Davipar/djev-dev.git 3ce907e6835212f27ee82b4cee9039198c4abe35 "$native_root/djev"
      native_python=python3
      uv pip install --system "$native_root/djev[tokenizer]"
      (cd "$native_root/djev" && "$native_python" -m runtime.install)
      start_process native bash -c "cd '$native_root/djev'; '$native_python' -m runtime.serve & upstream=\$!; DJEV_UPSTREAM=http://127.0.0.1:8001 '$native_python' -m djev --host 127.0.0.1 --port 8000 & api=\$!; trap 'kill \$upstream \$api 2>/dev/null || true; wait' TERM INT; wait -n \$upstream \$api; exit 1"
      native_pid=$started_pid
      wait_for_http native "$native_pid" http://127.0.0.1:8000/ready
      ;;
    djev-thinking)
      # DJeV Spark is a separate native /v1/systemone contract from djev-dev.
      # Keep its image/overlay/source trio pinned: the overlay is only valid
      # against the exact vLLM nightly named in the renderer below.
      clone_pinned https://github.com/mmastrac/djev-spark.git 1444f3e927f83ba508e5b28a4fd4fdd9ecd0976b "$native_root/djev-thinking"
      clone_pinned https://github.com/mmastrac/vllm.git 6591b093b29536dd070c6af3628b734025c53e23 "$native_root/djev-thinking-vllm"
      git -C "$native_root/djev-thinking-vllm" fetch --filter=blob:none https://github.com/vllm-project/vllm.git dee37d89115db4c94a820a79a78a7828e141c910
      local vllm_merge_base
      vllm_merge_base="$(git -C "$native_root/djev-thinking-vllm" merge-base dee37d89115db4c94a820a79a78a7828e141c910 HEAD)"
      git -C "$native_root/djev-thinking-vllm" diff --name-only "$vllm_merge_base" HEAD -- vllm >"$native_root/djev-thinking-vllm/changed.txt"
      git -C "$native_root/djev-thinking-vllm" diff --name-only --diff-filter=A "$vllm_merge_base" HEAD -- vllm >"$native_root/djev-thinking-vllm/added.txt"
      git -C "$native_root/djev-thinking-vllm" diff --quiet "$vllm_merge_base" dee37d89115db4c94a820a79a78a7828e141c910 -- $(<"$native_root/djev-thinking-vllm/changed.txt") || {
        echo "DJeV Spark overlay no longer matches vLLM nightly dee37d89115db4c94a820a79a78a7828e141c910" >&2
        return 1
      }
      bash "$native_root/djev-thinking/patches/link_cuda_headers.sh"
      python3 "$native_root/djev-thinking/patches/overlay_vllm.py" "$native_root/djev-thinking-vllm" dee37d89115db4c94a820a79a78a7828e141c910
      python3 "$native_root/djev-thinking/patches/raise_recompile_limit.py"
      cp "$native_root/djev-thinking/patches/spark_mem_trace.py" /usr/local/lib/python3.12/dist-packages/spark_mem_trace.py
      python3 "$native_root/djev-thinking/patches/worker_memory_cap.py"
      mkdir -p /opt/dgemma
      cp "$native_root/djev-thinking/server/structured_server.py" "$native_root/djev-thinking/server/playground.html" "$native_root/djev-thinking/server/walk.html" "$native_root/djev-thinking/server/cube.html" /opt/dgemma/
      hf download nvidia/diffusiongemma-26B-A4B-it-NVFP4 --revision ec4ff3df205028f4e81c954c2227f9312b3ec2ea --local-dir /workflow/models/djev-thinking
      start_process native env MODEL=/workflow/models/djev-thinking SERVED_NAME=djev-thinking CANVAS=64 MAX_SEQS=2 MAX_MODEL_LEN=32768 GPU_UTIL=0.85 STRUCTURED_PORT=8011 EXTRA_ARGS="--async-scheduling --kv-cache-dtype bfloat16" "$native_root/djev-thinking/entrypoint.sh"
      native_pid=$started_pid
      wait_for_http native "$native_pid" http://127.0.0.1:8011/health
      ;;
    jeff)
      clone_pinned https://github.com/logan-markewich/jeff.git 34b32f99a727c47b679adde33f4702a001e02979 "$native_root/jeff"
      uv sync --directory "$native_root/jeff"
      (cd "$native_root/jeff" && uv run hf download knowledgator/gliformer-large-v1 --revision d0a4e53d09cebe6bc963dd9be319d4279084bb2d --local-dir models/gliformer-large-v1)
      start_process native bash -c "cd '$native_root/jeff'; JEFF_MODEL='$native_root/jeff/models/gliformer-large-v1' JEFF_HOST=127.0.0.1 JEFF_PORT=8000 JEFF_MAX_LABELS=255 uv run jeff"
      native_pid=$started_pid
      wait_for_http native "$native_pid" http://127.0.0.1:8000/healthz
      ;;
    openjev-thinking|openjev-razorback16)
      # The renderer selects OpenJev's pinned public image.  Installing the
      # pinned source keeps the API code and the custom structured-read vLLM
      # revision together instead of substituting stock vLLM.
      clone_pinned https://github.com/razorback16/openjev.git 297a4efa843cca82b1e9989a3f50b5f1bdc49752 "$native_root/openjev"
      uv pip install --python "$(command -v python)" "$native_root/openjev"
      start_process native env -u OPENJEV_UPSTREAM \
        OPENJEV_HOST=127.0.0.1 OPENJEV_PORT=8080 \
        OPENJEV_MODEL=nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
        bash "$native_root/openjev/docker/entrypoint.sh"
      native_pid=$started_pid
      wait_for_http native "$native_pid" http://127.0.0.1:8080/v1/models
      ;;
    openjev-sglang)
      clone_pinned https://github.com/ekzhang/openjev-sglang.git f3e1678168b2e9a298bb47639444b428b661c4b2 "$native_root/openjev-sglang"
      uv sync --directory "$native_root/openjev-sglang" --frozen --no-default-groups --python 3.12
      start_process native env \
        OPENJEV_PROFILE=qwen36 \
        OPENJEV_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4 \
        OPENJEV_REVISION=1355db6a052410cfd62085d94b58866fd0f2c3c5 \
        OPENJEV_SERVED_MODEL_NAME=Qwen/Qwen3.6-35B-A3B \
        OPENJEV_FRONTEND=rust \
        "$native_root/openjev-sglang/.venv/bin/python" -m openjev serve \
          --host 127.0.0.1 --port 8080 --sglang-python /opt/sglang/bin/python
      native_pid=$started_pid
      wait_for_http native "$native_pid" http://127.0.0.1:8080/health
      ;;
    winnow-12b)
      clone_pinned https://github.com/EldanRing/winnow-inference.git 47ad31a338ebe2209e546b999c31f94853ddb75d "$native_root/winnow"
      (cd "$native_root/winnow" && python3 scripts/build.py --cuda-arch 89 --jobs 8)
      hf download EldanRing/Winnow-12B --revision b6ac22b0d51b69b18200acacb3fbdd98073fffe8 --include gguf/Winnow-12B-Q8_0.gguf --local-dir /workflow/models/winnow
      start_process native python3 "$native_root/winnow/scripts/serve.py" --model /workflow/models/winnow/gguf/Winnow-12B-Q8_0.gguf --text-only --host 127.0.0.1 --port 8091 --context 32768 --decision-context 32768
      native_pid=$started_pid
      wait_for_http native "$native_pid" http://127.0.0.1:8091/health
      ;;
    *)
      echo "unsupported native catalog model: $MODEL_KEY (expected djev, djev-thinking, jeff, openjev-thinking, openjev-razorback16, openjev-sglang, or winnow-12b)" >&2
      exit 2
      ;;
  esac
}

run_probe() {
  if ! curl -fsS "http://127.0.0.1:${compat_port}/health" >/dev/null; then
    echo "NATIVE_CATALOG_PROBE_HEALTH_FAILED model=$MODEL_KEY" >&2
    return 1
  fi
  local probe_status
  probe_status="$(curl -sS -o /workflow/probe-response.json -w '%{http_code}' "http://127.0.0.1:${compat_port}/v1/systemone" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_KEY\",\"state\":\"The light is on.\",\"questions\":{\"on\":{\"type\":\"noul\",\"instructions\":\"Is the light on?\"}}}")" || {
      echo "NATIVE_CATALOG_PROBE_TRANSPORT_FAILED model=$MODEL_KEY" >&2
      return 1
    }
  if [[ "$probe_status" != 200 ]]; then
    local detail
    detail="$(jq -r '(.detail // .error // "unknown") | tostring | .[0:400]' /workflow/probe-response.json 2>/dev/null || echo invalid-response)"
    echo "NATIVE_CATALOG_PROBE_HTTP_FAILED model=$MODEL_KEY status=$probe_status detail=$detail" >&2
    return 1
  fi
  jq -e '.answers.on.type == "noul" and (.answers.on.noul | type == "number")' /workflow/probe-response.json >/dev/null
  if [[ "$MODEL_KEY" == openjev-thinking || "$MODEL_KEY" == openjev-razorback16 || "$MODEL_KEY" == openjev-sglang || "$MODEL_KEY" == djev-thinking || "$MODEL_KEY" == winnow-12b || "$MODEL_KEY" == jeff ]]; then
    local wide_count=255
    if [[ "$MODEL_KEY" == djev-thinking ]]; then
      wide_count=27
    elif [[ "$MODEL_KEY" == winnow-12b || "$MODEL_KEY" == openjev-sglang ]]; then
      wide_count=65
    fi
    local wide_status
    wide_status="$(jq -nc --arg model "$MODEL_KEY" --argjson count "$wide_count" '{model:$model,state:"Choose the final option.",questions:{many:{type:"choice",instructions:"Choose one.",criteria:(reduce range(0;$count) as $i ({}; .["c"+($i|tostring)]="Option "+($i|tostring)))}}}' |
      curl -sS -o /workflow/probe-wide-response.json -w '%{http_code}' "http://127.0.0.1:${compat_port}/v1/systemone" -H 'Content-Type: application/json' -d @-)" || {
        echo "NATIVE_CATALOG_PROBE_WIDE_TRANSPORT_FAILED model=$MODEL_KEY" >&2
        return 1
      }
    if [[ "$wide_status" != 200 ]] || ! jq -e --argjson count "$wide_count" '.answers.many.type == "choice" and (.answers.many.probabilities | length) == $count' /workflow/probe-wide-response.json >/dev/null; then
      local detail
      detail="$(jq -r '(.detail // .error // "invalid-probabilities") | tostring | .[0:400]' /workflow/probe-wide-response.json 2>/dev/null || echo invalid-response)"
      echo "NATIVE_CATALOG_PROBE_WIDE_FAILED model=$MODEL_KEY status=$wide_status detail=$detail" >&2
      return 1
    fi
  fi
  echo "NATIVE_CATALOG_PROBE_COMPLETE model=$MODEL_KEY" >&2
}

summary_is_complete() {
  if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then
    jq -e '.requested_rows == 23900 and .successful_rows == 23900 and .error_rows == 0' "$1" >/dev/null
  else
    jq -e '.requested_rows == 23900 and (.successful_rows + .error_rows == 23900)' "$1" >/dev/null
  fi
}

run_suite() {
  bash "$chunk_helper" restore "$remote_result_dir" "$result_dir"
  if [[ -f "$result_dir/summary.json" ]] && summary_is_complete "$result_dir/summary.json"; then
    echo "NATIVE_CATALOG_SKIP model=$MODEL_KEY reason=complete" >&2
    return
  fi
  active_result_dir="$result_dir"
  active_remote_dir="$remote_result_dir"
  bash "$chunk_helper" loop "$remote_result_dir" "$result_dir" &
  chunk_pid=$!
  (cd "$source_root" && uv run python -m decision_bench.cli run-jev-server \
    "$source_root/task_specs/decisionbench-dev.toml" "$result_dir" \
    --project-root "$source_root" --base-url "http://127.0.0.1:${compat_port}" \
    --model "$MODEL_KEY" --concurrency 8) &
  local evaluation_pid=$!
  while kill -0 "$evaluation_pid" 2>/dev/null; do
    for watched in "$native_pid:native" "$compat_pid:compat"; do
      local pid="${watched%%:*}" name="${watched##*:}"
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "NATIVE_CATALOG_${name}_DIED_DURING_EVALUATION model=$MODEL_KEY" >&2
        show_log "$name"
        kill "$evaluation_pid" 2>/dev/null || true
        wait "$evaluation_pid" 2>/dev/null || true
        return 1
      fi
    done
    sleep 5
  done
  wait "$evaluation_pid"
  summary_is_complete "$result_dir/summary.json"
  stop_process "$chunk_pid"
  chunk_pid=
  bash "$chunk_helper" publish "$remote_result_dir" "$result_dir"
  active_result_dir=
  active_remote_dir=
  echo "NATIVE_CATALOG_COMPLETE model=$MODEL_KEY rows=23900" >&2
}

if [[ ! -f /workflow/decision-bench-runtime.tgz ]]; then
  hf buckets cp "$SOURCE_ARCHIVE" /workflow/decision-bench-runtime.tgz
fi
tar -xzf /workflow/decision-bench-runtime.tgz -C "$source_root" --strip-components=1
uv sync --directory "$source_root" --extra hf
install_compat_server
start_native_server
start_compat_server

case "${RUN_MODE:-suite}" in
  probe) run_probe ;;
  suite)
    if [[ "$MODEL_KEY" == openjev-sglang || "$MODEL_KEY" == jeff ]]; then
      run_probe
    fi
    run_suite
    ;;
  *) echo "RUN_MODE must be probe or suite" >&2; exit 2 ;;
esac
