#!/bin/bash
# Download HuggingFace models to scratch cache.
# Usage:
#   bash src/commit_utils/download_models.sh             # download default sycophancy models
#   bash src/commit_utils/download_models.sh large       # include larger models (27B+)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/set_env_vars.sh"

MODE="${1:-default}"

# Core sycophancy reproduction models (~41 GB)
DEFAULT_MODELS=(
    "google/gemma-3-12b-it"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Optional extension models (~60 GB additional)
LARGE_MODELS=(
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "google/gemma-3-27b-it"
)

if [ "$MODE" = "large" ]; then
    MODELS=("${DEFAULT_MODELS[@]}" "${LARGE_MODELS[@]}")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

echo "=== Downloading models to ${AUTOINTERP_HF_HOME} ==="
echo "Models: ${MODELS[*]}"
echo ""

# Write download script to scratch (container can't see /tmp)
DOWNLOAD_SCRIPT="${AUTOINTERP_SCRATCH}/download_models_tmp.py"
cat > "$DOWNLOAD_SCRIPT" << PYEOF
from huggingface_hub import snapshot_download
import os, sys
cache = os.environ.get('HF_HOME')
models = sys.argv[1:]
for m in models:
    print(f'Downloading {m}...', flush=True)
    snapshot_download(m, cache_dir=cache)
    print(f'Done: {m}', flush=True)
PYEOF

srun \
    --account="${CSCS_ACCOUNT}" \
    --environment="${AUTOINTERP_EDF}" \
    --ntasks=1 \
    python3 "$DOWNLOAD_SCRIPT" "${MODELS[@]}"

rm -f "$DOWNLOAD_SCRIPT"
echo ""
echo "=== All downloads complete ==="
