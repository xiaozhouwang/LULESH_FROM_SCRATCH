#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPU_BIN="${CPU_BIN:-${ROOT_DIR}/lulesh2.0}"
GPU_BIN="${GPU_BIN:-${ROOT_DIR}/lulesh-gpu-opt/lulesh-cuda/lulesh_gpu}"

LOG_CYCLES="${LOG_CYCLES:-10}"
LOG_CYCLE_STRIDE="${LOG_CYCLE_STRIDE:-1}"
SIZE="${SIZE:-20}"
ITERS="${ITERS:-${LOG_CYCLES}}"
OMP_THREADS="${OMP_NUM_THREADS:-1}"

LOG_PRE="${LOG_PRE:-1}"
LOG_SUBSTEPS="${LOG_SUBSTEPS:-1}"
LOG_FIELDS="${LOG_FIELDS:-}"
COMPARE_ARGS="${COMPARE_ARGS:-}"

CPU_LOG_ROOT="${CPU_LOG_ROOT:-${ROOT_DIR}/benchmarks/logs-multi/cycles${LOG_CYCLES}-s${LOG_CYCLE_STRIDE}}"
GPU_LOG_ROOT="${GPU_LOG_ROOT:-${ROOT_DIR}/benchmarks/logs-gpu-multi/cycles${LOG_CYCLES}-s${LOG_CYCLE_STRIDE}}"

if [[ ! -x "${CPU_BIN}" ]]; then
  echo "CPU binary not found: ${CPU_BIN}" >&2
  exit 1
fi
if [[ ! -x "${GPU_BIN}" ]]; then
  echo "GPU binary not found: ${GPU_BIN}" >&2
  exit 1
fi

export LULESH_LOG_ENABLE=1
export LULESH_LOG_PRE="${LOG_PRE}"
export LULESH_LOG_SUBSTEPS="${LOG_SUBSTEPS}"
export LULESH_LOG_CYCLES="${LOG_CYCLES}"
export LULESH_LOG_CYCLE_STRIDE="${LOG_CYCLE_STRIDE}"
if [[ -n "${LOG_FIELDS}" ]]; then
  export LULESH_LOG_FIELDS="${LOG_FIELDS}"
else
  unset LULESH_LOG_FIELDS
fi

echo "CPU log root: ${CPU_LOG_ROOT}"
echo "GPU log root: ${GPU_LOG_ROOT}"
echo "Running CPU: size=${SIZE} iters=${ITERS}"
OMP_NUM_THREADS="${OMP_THREADS}" LULESH_LOG_ROOT="${CPU_LOG_ROOT}" \
  "${CPU_BIN}" -s "${SIZE}" -i "${ITERS}"

echo "Running GPU: size=${SIZE} iters=${ITERS}"
OMP_NUM_THREADS="${OMP_THREADS}" LULESH_LOG_ROOT="${GPU_LOG_ROOT}" \
  "${GPU_BIN}" -s "${SIZE}" -i "${ITERS}"

echo "Comparing logs..."
python3 "${ROOT_DIR}/benchmarks/compare_logs.py" \
  --cpu "${CPU_LOG_ROOT}" \
  --gpu "${GPU_LOG_ROOT}" \
  --precision double \
  ${COMPARE_ARGS}
