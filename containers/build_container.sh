#!/bin/bash
set -euo pipefail

container="${1:-autointerp}"

: "${AUTOINTERP_CONTAINER_TAG:=autointerp:${container}}"
: "${AUTOINTERP_CONTAINERS_DIR:=${SCRATCH:-$PWD}/containers}"

mkdir -p "${AUTOINTERP_CONTAINERS_DIR}"

podman build -t "${AUTOINTERP_CONTAINER_TAG}" -f containers/Containerfile containers/

enroot import \
    -o "${AUTOINTERP_CONTAINERS_DIR}/${container}.sqsh" \
    "podman://${AUTOINTERP_CONTAINER_TAG}"

echo "Built ${AUTOINTERP_CONTAINERS_DIR}/${container}.sqsh"
