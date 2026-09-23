#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 TASK_NAME RESULT_NAME MODEL_KEY RUN_MODE GPU_MEMORY_GB" >&2
  exit 2
fi

task_name="$1"
result_name="$2"
model_key="$3"
run_mode="$4"
gpu_memory="$5"
template="$(cd "$(dirname "$0")" && pwd)/decisionbench_native_catalog_base.dstack.yml"
gpu_names="[L40S, A100, H100, H200, B200]"

for value in "$task_name" "$result_name" "$model_key"; do
  [[ "$value" =~ ^[A-Za-z0-9._/-]+$ ]] || {
    echo "task, result, and model identifiers must use safe path characters" >&2
    exit 2
  }
done
[[ "$run_mode" == suite || "$run_mode" == probe ]] || {
  echo "RUN_MODE must be suite or probe" >&2
  exit 2
}
[[ "$gpu_memory" =~ ^[1-9][0-9]*$ ]] || {
  echo "GPU memory must be a positive integer number of GB" >&2
  exit 2
}

case "$model_key" in
  djev)
    image="vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910"
    ;;
  djev-thinking)
    # mmastrac/djev-spark@1444f3e requires this exact vLLM base before its
    # mmastrac/vllm@6591b093 structured-reads overlay is applied at startup.
    image="vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910"
    ;;
  jeff)
    image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime"
    ;;
  winnow-12b)
    # setup.py compiles llama.cpp, so this must include nvcc rather than the
    # smaller PyTorch runtime image used by Jeff.
    image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
    ;;
  openjev-thinking)
    image="razorback16/openjev:0.3.0"
    ;;
  openjev-razorback16)
    image="razorback16/openjev:0.3.0"
    ;;
  openjev-sglang)
    image="lmsysorg/sglang:v0.5.19-cu130"
    gpu_names="[B200]"
    ;;
  *)
    echo "MODEL_KEY must be djev, djev-thinking, jeff, openjev-thinking, openjev-razorback16, openjev-sglang, or winnow-12b" >&2
    exit 2
    ;;
esac

sed \
  -e "s/__TASK_NAME__/$task_name/g" \
  -e "s/__RUN_NAME__/$result_name/g" \
  -e "s/__MODEL_KEY__/$model_key/g" \
  -e "s/__RUN_MODE__/$run_mode/g" \
  -e "s|__IMAGE__|$image|g" \
  -e "s|__GPU_NAMES__|$gpu_names|g" \
  -e "s/__GPU_MEMORY__/$gpu_memory/g" \
  "$template"

