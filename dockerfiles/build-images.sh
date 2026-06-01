#!/bin/bash
set -e

DOCKER_COMMAND="docker"
XAAS_IMAGE_PREFIX="docker.io/spcleth/xaas:"

CONTEXT_PATH=.

DEBIAN_VERSION=13
LLVM_VERSION=19

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

for CUDA_VERSION in "13.3"; do
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
