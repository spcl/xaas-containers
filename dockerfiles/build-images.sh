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

print_usage() {
  echo "Usage: build-images.sh [OPTION]... [TARGETS]...

Options:
  --build       Build images.
  --push        Push images to the remote docker repository.

Targets:
  base          The base builder/runtime layers for the xaas system.
  layer-cuda    The dependency layers for the CUDA compiler and runtime.
  layer-fftw3   The dependency layers for the FFTW3 library.
  layer-mpich   The dependency layers for the MPICH library.
  layer-oneapi  The dependency layers for the OneAPI compiler and runtime.

  If no targets are specified, all targets will be built." >&2
  exit 1
}

should_build_images=false
should_push_images=false

any_target_specified=false
target_base=false
target_layer_cuda=false
target_layer_fftw3=false
target_layer_mpich=false
target_layer_oneapi=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) should_build_images=true;;
    --push)  should_push_images=true;;
    -*)
      echo "Unknown option: $1" >&2
      print_usage
      ;;

    base) any_target_specified=true; target_base=true;;
    layer-cuda) any_target_specified=true; target_layer_cuda=true;;
    layer-fftw3) any_target_specified=true; target_layer_fftw3=true;;
    layer-mpich) any_target_specified=true; target_layer_mpich=true;;
    layer-oneapi) any_target_specified=true; target_layer_oneapi=true;;
    *)
      echo "Unknown target: $1" >&2
      print_usage
      ;;
  esac
  shift
done

if [ ! "${should_build_images}" = true ] && [ ! "${should_push_images}" = true ]; then
  echo "Neither --build nor --push was specified, defaulting to --build" >&2
  should_build_images=true
fi

if [ ! "${any_target_specified}" = true ]; then
  echo "No targets were specified, defaulting to all" >&2
  target_base=true
  target_layer_cuda=true
  target_layer_fftw3=true
  target_layer_mpich=true
  target_layer_oneapi=true
fi

BUILD_ARGS=(
  --build-arg=XAAS_IMAGE_PREFIX="${XAAS_IMAGE_PREFIX}"
  --build-arg=DEBIAN_VERSION="${DEBIAN_VERSION}"
  --build-arg=LLVM_VERSION="${LLVM_VERSION}"
  --platform="linux/amd64"
)

BUILT_IMAGE_TAGS=()

build_image() {
  local args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--tag)
        # save tag, then propagate the arguments
        BUILT_IMAGE_TAGS+=("$2")
        args+=("$1" "$2")
        shift 2
        ;;
      --tag=*)
        # save tag, then propagate the arguments
        BUILT_IMAGE_TAGS+=("${1#--tag=}")
        args+=("$1")
        shift
        ;;
      *)
        args+=("$1") # save positional arg
        shift # past argument
        ;;
    esac
  done

  if [ "${should_build_images}" = true ]; then
    "${DOCKER_COMMAND}" build "${args[@]}"
  fi
}

# base layers

if [ "${target_base}" = true ]; then
  build_image "${BUILD_ARGS[@]}" \
      --tag="${XAAS_IMAGE_PREFIX}base-debian${DEBIAN_VERSION}" \
      --file="Dockerfile.base-debian${DEBIAN_VERSION}" "$CONTEXT_PATH"

  build_image "${BUILD_ARGS[@]}" \
      --tag="${XAAS_IMAGE_PREFIX}compiler-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" \
      --file="Dockerfile.compiler-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" "$CONTEXT_PATH"

  build_image "${BUILD_ARGS[@]}" \
      --tag="${XAAS_IMAGE_PREFIX}builder-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" \
      --file="Dockerfile.builder-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" "$CONTEXT_PATH"

  build_image "${BUILD_ARGS[@]}" \
      --tag="${XAAS_IMAGE_PREFIX}runner-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" \
      --file="Dockerfile.runner-debian${DEBIAN_VERSION}-llvm${LLVM_VERSION}" "$CONTEXT_PATH"
fi

# dependency layers

if [ "${target_layer_cuda}" = true ]; then
  for CUDA_VERSION in "13.1" "13.3"; do
    build_image "${BUILD_ARGS[@]}" \
        --build-arg=CUDA_VERSION="${CUDA_VERSION}" \
        --tag="${XAAS_IMAGE_PREFIX}layer-cuda${CUDA_VERSION}" \
        --file="layers/Dockerfile.cuda" "$CONTEXT_PATH"
  done
fi

if [ "${target_layer_mpich}" = true ]; then
  for MPICH_VERSION in "4.3.0"; do
    build_image "${BUILD_ARGS[@]}" \
        --build-arg=MPICH_VERSION="${MPICH_VERSION}" \
        --tag="${XAAS_IMAGE_PREFIX}layer-mpich${MPICH_VERSION}" \
        --file="layers/Dockerfile.mpich" "$CONTEXT_PATH"
  done
fi

if [ "${target_layer_oneapi}" = true ]; then
  for ONEAPI_VERSION in "2025.0"; do
    build_image "${BUILD_ARGS[@]}" \
        --build-arg=ONEAPI_VERSION="${ONEAPI_VERSION}" \
        --tag="${XAAS_IMAGE_PREFIX}layer-oneapi${ONEAPI_VERSION}" \
        --file="layers/Dockerfile.oneapi" "$CONTEXT_PATH"
  done
fi

if [ "${target_layer_fftw3}" = true ]; then
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

      build_image "${BUILD_ARGS[@]}" \
          --build-arg=FFTW3_VERSION="${FFTW3_VERSION}" \
          --build-arg=FFTW3_SIMD_BUILD_ARGS="${FFTW3_SIMD_BUILD_ARGS}" \
          --tag="${XAAS_IMAGE_PREFIX}layer-fftw${FFTW3_VERSION}-${FFTW3_SIMD_TAG}" \
          --file="layers/Dockerfile.fftw3" "$CONTEXT_PATH"
    done
  done
fi

# push the resulting tags if requested

if [ "${should_push_images}" = true ]; then
  echo "Pushing ${#BUILT_IMAGE_TAGS[@]} images..." >&2
  for tag in "${BUILT_IMAGE_TAGS[@]}"; do
    "${DOCKER_COMMAND}" push "${tag}"
  done
fi
