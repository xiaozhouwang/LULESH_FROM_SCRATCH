#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/lulesh-cuda/build}"
NVCC="${NVCC:-nvcc}"
CXX="${CXX:-g++}"
ARCH="${ARCH:-sm_89}"

mkdir -p "${BUILD_DIR}"

CPU_FLAGS="-O3 -DUSE_MPI=0 -fopenmp -I${ROOT_DIR}"
CUDA_FLAGS="-O3 -std=c++14 -arch=${ARCH} -DUSE_MPI=0 -I${ROOT_DIR}"

${CXX} ${CPU_FLAGS} -c "${ROOT_DIR}/lulesh-util.cc" -o "${BUILD_DIR}/lulesh-util.o"
${CXX} ${CPU_FLAGS} -c "${ROOT_DIR}/lulesh-init.cc" -o "${BUILD_DIR}/lulesh-init.o"
${CXX} ${CPU_FLAGS} -c "${ROOT_DIR}/lulesh-comm.cc" -o "${BUILD_DIR}/lulesh-comm.o"
${CXX} ${CPU_FLAGS} -c "${ROOT_DIR}/lulesh-viz.cc" -o "${BUILD_DIR}/lulesh-viz.o"

${NVCC} ${CUDA_FLAGS} -Xcompiler -fopenmp \
  "${ROOT_DIR}/lulesh-cuda/lulesh_cuda.cu" \
  "${BUILD_DIR}/lulesh-util.o" \
  "${BUILD_DIR}/lulesh-init.o" \
  "${BUILD_DIR}/lulesh-comm.o" \
  "${BUILD_DIR}/lulesh-viz.o" \
  -o "${BUILD_DIR}/lulesh_cuda"

echo "Built ${BUILD_DIR}/lulesh_cuda"
