#!/bin/bash
set -euo pipefail

container="${1:-autointerp}"

: "${AUTOINTERP_CONTAINER_TAG:=autointerp:${container}}"
: "${AUTOINTERP_CONTAINERS_DIR:=${SCRATCH:-$PWD}/autointerp/containers}"
: "${AUTOINTERP_PODMAN_SHM_DIR:=/dev/shm/${USER}/autointerp-podman}"
: "${AUTOINTERP_PODMAN_CONF_DIR:=${XDG_CONFIG_HOME:-$HOME/.config}/containers}"
: "${AUTOINTERP_PODMAN_STORAGE_CONF:=${AUTOINTERP_PODMAN_CONF_DIR}/storage.conf}"

mkdir -p "${AUTOINTERP_CONTAINERS_DIR}" "${AUTOINTERP_PODMAN_CONF_DIR}"
mkdir -p "${AUTOINTERP_PODMAN_SHM_DIR}/root" "${AUTOINTERP_PODMAN_SHM_DIR}/runroot"

if [ ! -f "${AUTOINTERP_PODMAN_STORAGE_CONF}" ]; then
    cat > "${AUTOINTERP_PODMAN_STORAGE_CONF}" <<EOF
[storage]
driver = "overlay"
runroot = "${AUTOINTERP_PODMAN_SHM_DIR}/runroot"
graphroot = "${AUTOINTERP_PODMAN_SHM_DIR}/root"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs-1.13"
EOF
    echo "Created ${AUTOINTERP_PODMAN_STORAGE_CONF}"
else
    echo "Using existing Podman storage config: ${AUTOINTERP_PODMAN_STORAGE_CONF}"
fi

export CONTAINERS_STORAGE_CONF="${AUTOINTERP_PODMAN_STORAGE_CONF}"

echo "Building ${AUTOINTERP_CONTAINER_TAG}..."
podman build -t "${AUTOINTERP_CONTAINER_TAG}" -f containers/Containerfile containers/

echo "Importing as .sqsh (must complete before this allocation ends)..."
enroot import -x mount \
    -o "${AUTOINTERP_CONTAINERS_DIR}/${container}.sqsh" \
    "podman://${AUTOINTERP_CONTAINER_TAG}"

echo "Built ${AUTOINTERP_CONTAINERS_DIR}/${container}.sqsh"
