#!/bin/bash

set_default() {
    local var_name="${1:-}"
    local default_value="${2:-}"
    local current_value
    eval "current_value=\"\${$var_name:-}\""
    if [ -z "$current_value" ] || [ "$current_value" = "UNDEFINED" ]; then
        export "$var_name"="$default_value"
    fi
}

# Core paths
set_default AUTOINTERP_ROOT "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set_default AUTOINTERP_SCRATCH "${SCRATCH:-/iopsstor/scratch/cscs/${USER}/autointerp}"
set_default AUTOINTERP_RESULTS_DIR "${AUTOINTERP_SCRATCH}/results"
set_default AUTOINTERP_CONTAINERS_DIR "${AUTOINTERP_SCRATCH}/containers"

# Container
set_default AUTOINTERP_CONTAINER_NAME "autointerp"
set_default AUTOINTERP_CONTAINER_IMAGE "${AUTOINTERP_CONTAINERS_DIR}/${AUTOINTERP_CONTAINER_NAME}.sqsh"
set_default AUTOINTERP_EDF "${AUTOINTERP_CONTAINERS_DIR}/autointerp.toml"

# HuggingFace cache (pre-populated on scratch before running jobs)
set_default AUTOINTERP_HF_HOME "${AUTOINTERP_SCRATCH}/hf_cache"
set_default HF_HOME "${AUTOINTERP_HF_HOME}"
set_default HF_TOKEN "${HF_TOKEN:-}"

# Slurm / scheduler
set_default AUTOINTERP_JOB_SCHEDULER "slurm_clariden"
set_default CSCS_ACCOUNT ""

# Experiment settings
set_default AUTOINTERP_TIME_LIMIT "2"
set_default AUTOINTERP_EXPERIMENT_NAME ""

# Validation
if [ -z "${CSCS_ACCOUNT}" ]; then
    echo "WARNING: CSCS_ACCOUNT is not set. Set it with: export CSCS_ACCOUNT='your-project-account'"
fi

if [ ! -f "${AUTOINTERP_EDF}" ]; then
    echo "WARNING: EDF not found at ${AUTOINTERP_EDF}. Generate it with:"
    echo "  envsubst < containers/autointerp.toml.template > \${AUTOINTERP_EDF}"
fi

if [ ! -f "${AUTOINTERP_CONTAINER_IMAGE}" ]; then
    echo "WARNING: Container image not found at ${AUTOINTERP_CONTAINER_IMAGE}. Build with:"
    echo "  bash containers/build_container.sh"
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${OPENCODE_API_KEY:-}" ]; then
    echo "WARNING: No agent API keys set. Export at least one of:"
    echo "  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, OPENCODE_API_KEY"
fi

mkdir -p "${AUTOINTERP_RESULTS_DIR}"
