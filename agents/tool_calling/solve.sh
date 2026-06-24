#!/bin/bash
set -euo pipefail

: "${AUTOINTERP_ROOT:?AUTOINTERP_ROOT must be set}"
: "${AGENT_CONFIG:?AGENT_CONFIG must be set}"
: "${TASK:?TASK must be set}"
: "${MODEL:?MODEL must be set}"

python3 "${AUTOINTERP_ROOT}/src/tool_calling/run_agent.py" \
    --task "${TASK}" \
    --model "${AGENT_CONFIG}" \
    --model-path "${MODEL}" \
    --output-dir "$(pwd)"
