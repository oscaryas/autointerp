#!/bin/bash
set -e

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi
echo "======================="

# Print environment
echo "=== Environment ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HF_HOME: $HF_HOME"
echo "Working directory: $(pwd)"
echo "==================="

# Execute command
exec "$@"
