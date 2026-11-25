# Shared environment for AggMath jobs on Unity. Source it from a job script.
# shellcheck shell=bash

export AGGMATH_ROOT="${AGGMATH_ROOT:-/work/pi_hzamani_umass_edu/runze/AggMath}"
export AGGMATH_WS="${AGGMATH_WS:-/scratch3/workspace/runzefan_umass_edu-aggmath}"
CONDA_ROOT="${CONDA_ROOT:-/work/pi_hzamani_umass_edu/runze/miniconda3}"
AGGMATH_CONDA_ENV="${AGGMATH_CONDA_ENV:-aggmath}"

# conda's activate hooks read unset variables, so relax nounset around them.
set +u
# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${AGGMATH_CONDA_ENV}"
set -u

# Only use the aggmath env, never ~/.local site-packages.
export PYTHONNOUSERSITE=1

# Models and benchmark datasets are pre-fetched into the workspace HF cache
# (Qwen3.5-9B links to /datasets/ai/qwen3/hub), so jobs run offline.
export HF_HOME="${AGGMATH_WS}/hf_home"
export HF_HUB_CACHE="${AGGMATH_WS}/hf_cache/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
# LiteLLM downloads a model-price map at import time unless told to use its local copy.
export LITELLM_LOCAL_MODEL_COST_MAP=True

# Keep compile caches on scratch instead of $HOME.
export XDG_CACHE_HOME="${AGGMATH_WS}/cache"
export VLLM_CACHE_ROOT="${AGGMATH_WS}/cache/vllm"
export TRITON_CACHE_DIR="${AGGMATH_WS}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${AGGMATH_WS}/cache/torchinductor"

export GLOO_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${AGGMATH_ROOT}" || return 1
mkdir -p logs/slurm

qwen35_repo() {
  case "$1" in
    2b) echo "Qwen/Qwen3.5-2B" ;;
    4b) echo "Qwen/Qwen3.5-4B" ;;
    9b) echo "Qwen/Qwen3.5-9B" ;;
    *)
      echo "Unknown model size '$1' (expected 2b, 4b or 9b)." >&2
      return 1
      ;;
  esac
}

# Array jobs from slurm/submit.sh: TASKS="9b:hmmt24 4b:aime24 ..." and
# SLURM_ARRAY_TASK_ID picks one size:benchmark entry.
resolve_array_task() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${TASKS:-}" ]]; then
    local -a tasks
    read -r -a tasks <<< "${TASKS}"
    local task="${tasks[${SLURM_ARRAY_TASK_ID}]}"
    SIZE="${task%%:*}"
    BENCH="${task#*:}"
    export SIZE BENCH
  fi
}

# Cancel this job if its GPU stays idle, so it never holds an allocation while
# doing nothing (Unity flags GPUs idle for over an hour and kills them at two).
# Utilization is sampled once a minute after a grace period that covers model
# loading and compilation. Call it from GPU jobs after the job has started.
start_gpu_watchdog() {
  local max_idle_minutes="${1:-20}" grace_minutes="${2:-20}"
  (
    sleep $((grace_minutes * 60))
    idle=0
    while true; do
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2> /dev/null | sort -n | tail -1)
      if [[ -n "${util}" && "${util}" -eq 0 ]]; then
        idle=$((idle + 1))
      else
        idle=0
      fi
      if ((idle >= max_idle_minutes)); then
        echo "[gpu-watchdog] GPU idle for ${idle} min; cancelling job ${SLURM_JOB_ID:-}" >&2
        scancel "${SLURM_JOB_ID}"
        exit 0
      fi
      sleep 60
    done
  ) &
  echo "gpu watchdog: cancel after ${max_idle_minutes} idle minutes (grace ${grace_minutes} min)"
}

print_job_info() {
  echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none}"
  module list 2>&1 || true
  which python
  python --version
  python -c 'import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__)'
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2> /dev/null || true
}
