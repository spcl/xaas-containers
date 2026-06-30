#!/bin/bash
set -e
set -u

XAAS_SYSTEM_REPO="docker.io/spcleth/xaas"
XAAS_SYSTEM_VERSION="1_rc1"

DEBIAN_VERSION=13
LLVM_VERSION=19

DOCKER_COMMAND="docker"
XAAS_IMAGE_PREFIX="${XAAS_SYSTEM_REPO}:${XAAS_SYSTEM_VERSION}-"

CONTEXT_PATH=.

BUILD_ARGS=(
  --build-arg=XAAS_IMAGE_PREFIX="${XAAS_IMAGE_PREFIX}"
  --build-arg=DEBIAN_VERSION="${DEBIAN_VERSION}"
  --build-arg=LLVM_VERSION="${LLVM_VERSION}"
  --platform="linux/amd64"
)

"$DOCKER_COMMAND" build "${BUILD_ARGS[@]}" \
    --tag="${XAAS_IMAGE_PREFIX}base-debian${DEBIAN_VERSION}" \
    --file="Dockerfile.base-debian${DEBIAN_VERSION}" "$CONTEXT_PATH"

"$DOCKER_COMMAND" build "${BUILD_ARGS[@]}" \
    --tag="${XAAS_IMAGE_PREFIX}compiler-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" \
    --file="Dockerfile.compiler-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" "$CONTEXT_PATH"

"$DOCKER_COMMAND" build "${BUILD_ARGS[@]}" \
    --tag="${XAAS_IMAGE_PREFIX}builder-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" \
    --file="Dockerfile.builder-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" "$CONTEXT_PATH"

"$DOCKER_COMMAND" build "${BUILD_ARGS[@]}" \
    --tag="${XAAS_IMAGE_PREFIX}runner-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" \
    --file="Dockerfile.runner-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" "$CONTEXT_PATH"

# build layers

for CUDA_VERSION in "13.1" "13.3"; do
  "$DOCKER_COMMAND" build "${BUILD_ARGS[@]}" \
      --build-arg=CUDA_VERSION="${CUDA_VERSION}" \
      --tag="${XAAS_IMAGE_PREFIX}layer-cuda${CUDA_VERSION}" \
      --file="layers/Dockerfile.cuda" "$CONTEXT_PATH"
done

for MPICH_VERSION in "4.3.0"; do
  "$DOCKER_COMMAND" build "${BUILD_ARGS[@]}" \
      --build-arg=MPICH_VERSION="${MPICH_VERSION}" \
      --tag="${XAAS_IMAGE_PREFIX}layer-mpich${MPICH_VERSION}" \
      --file="layers/Dockerfile.mpich" "$CONTEXT_PATH"
done

for ONEAPI_VERSION in "2025.0"; do
  "$DOCKER_COMMAND" build "${BUILD_ARGS[@]}" \
      --build-arg=ONEAPI_VERSION="${ONEAPI_VERSION}" \
      --tag="${XAAS_IMAGE_PREFIX}layer-oneapi${ONEAPI_VERSION}" \
      --file="layers/Dockerfile.oneapi" "$CONTEXT_PATH"
done

for FFTW3_VERSION in "3.3.10"; do
  FFTW3_TARGETS=(
    "SSE2,--enable-sse2"
    "SSE4.1,--enable-sse2"
    "AVX_256,--enable-sse2 --enable-avx --enable-avx2"
    "AVX2_128,--enable-sse2 --enable-avx --enable-avx2"
    "AVX2_256,--enable-sse2 --enable-avx --enable-avx2"
    "AVX_512,--enable-sse2 --enable-avx --enable-avx2 --enable-avx512"
  )
  for target in "${FFTW3_TARGETS[@]}"; do
    IFS=',' read FFTW3_SIMD_TAG FFTW3_SIMD_BUILD_ARGS <<< "${target}"

    "$DOCKER_COMMAND" build "${BUILD_ARGS[@]}" \
        --build-arg=FFTW3_VERSION="${FFTW3_VERSION}" \
        --build-arg=FFTW3_SIMD_BUILD_ARGS="${FFTW3_SIMD_BUILD_ARGS}" \
        --tag="${XAAS_IMAGE_PREFIX}layer-fftw${FFTW3_VERSION}-${FFTW3_SIMD_TAG}" \
        --file="layers/Dockerfile.fftw3" "$CONTEXT_PATH"
  done
done
