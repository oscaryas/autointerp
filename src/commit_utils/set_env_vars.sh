#!/bin/bash

# Helper function: sets variable to default if unset or "UNDEFINED"
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
set_default AUTOINTERP_RESULTS_DIR "${AUTOINTERP_ROOT}/results"
set_default AUTOINTERP_CONTAINERS_DIR "${AUTOINTERP_ROOT}/containers"

# Container configuration
set_default AUTOINTERP_CONTAINER_IMAGE "autointerp:latest"
# For Runpod, use your Docker Hub image:
# set_default AUTOINTERP_CONTAINER_IMAGE "yourusername/autointerp:latest"

# Runpod configuration
set_default RUNPOD_GPU_TYPE "NVIDIA A100"  # Options: "NVIDIA A100", "NVIDIA A100 80GB", "NVIDIA RTX A6000", etc.
set_default RUNPOD_GPU_COUNT "1"
set_default RUNPOD_DISK_SIZE "50"  # GB
set_default RUNPOD_CLOUD_TYPE "SECURE"  # Options: "SECURE", "COMMUNITY"
set_default RUNPOD_REGION "US"  # Optional: specify region

# Model cache
set_default HF_HOME "$HOME/.cache/huggingface"
set_default HF_TOKEN "${HF_TOKEN:-}"

# Experiment configuration
set_default AUTOINTERP_EXPERIMENT_NAME ""
set_default AUTOINTERP_TIME_LIMIT "2"  # hours

# Python environment
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Logging
set_default AUTOINTERP_LOG_LEVEL "INFO"

# Validate required API keys
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "WARNING: RUNPOD_API_KEY is not set. You will not be able to submit jobs."
    echo "Set it with: export RUNPOD_API_KEY='your-key'"
fi

# Check for at least one agent API key
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
    echo "WARNING: No agent API keys set. You need at least one of:"
    echo "  - ANTHROPIC_API_KEY (for Claude)"
    echo "  - OPENAI_API_KEY (for GPT/Codex)"
    echo "  - GEMINI_API_KEY (for Gemini)"
fi

# Create results directory if it doesn't exist
mkdir -p "${AUTOINTERP_RESULTS_DIR}"

# Print configuration (for debugging)
if [ "${AUTOINTERP_LOG_LEVEL}" = "DEBUG" ]; then
    echo "=== AutoInterp Configuration ==="
    echo "AUTOINTERP_ROOT: ${AUTOINTERP_ROOT}"
    echo "AUTOINTERP_RESULTS_DIR: ${AUTOINTERP_RESULTS_DIR}"
    echo "AUTOINTERP_CONTAINER_IMAGE: ${AUTOINTERP_CONTAINER_IMAGE}"
    echo "RUNPOD_GPU_TYPE: ${RUNPOD_GPU_TYPE}"
    echo "RUNPOD_GPU_COUNT: ${RUNPOD_GPU_COUNT}"
    echo "HF_HOME: ${HF_HOME}"
    echo "================================"
fi
