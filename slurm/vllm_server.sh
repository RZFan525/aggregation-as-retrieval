# Run the repo's scripts/serve_qwen35.sh inside a job. Source after slurm/env.sh.
# shellcheck shell=bash

# start_vllm_server <hf_repo> <max_model_len> <log_file>
# Only MODEL, PORT, DATA_PARALLEL_SIZE and MAX_MODEL_LEN are set; every other
# server option keeps the default of scripts/serve_qwen35.sh.
start_vllm_server() {
  local model="$1" max_model_len="$2"
  VLLM_LOG="$3"
  # Ask the OS for a free port so jobs sharing a node do not collide.
  VLLM_PORT="$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
  VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
  MODEL="${model}" \
    PORT="${VLLM_PORT}" \
    DATA_PARALLEL_SIZE="${SLURM_GPUS_ON_NODE:-1}" \
    MAX_MODEL_LEN="${max_model_len}" \
    bash scripts/serve_qwen35.sh > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  trap stop_vllm_server EXIT
  echo "vLLM pid=${VLLM_PID} model=${model} max_model_len=${max_model_len} url=${VLLM_BASE_URL} log=${VLLM_LOG}"
}

wait_for_vllm_server() {
  local timeout="${1:-1800}" waited=0
  until curl -fsS "${VLLM_BASE_URL}/models" > /dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2> /dev/null; then
      echo "vLLM exited before it became ready. Last log lines:" >&2
      tail -n 50 "${VLLM_LOG}" >&2
      return 1
    fi
    if ((waited >= timeout)); then
      echo "vLLM was not ready after ${timeout}s." >&2
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "vLLM ready after ${waited}s"
}

stop_vllm_server() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2> /dev/null; then
    kill "${VLLM_PID}" 2> /dev/null || true
    wait "${VLLM_PID}" 2> /dev/null || true
  fi
}
