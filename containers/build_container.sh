#!/bin/bash
set -e

echo "Building AutoInterp container..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build Docker image
docker build -t autointerp:latest .

echo "Container built successfully!"
echo ""
echo "To test the container locally:"
echo "  docker run --gpus all -it --rm autointerp:latest"
echo ""
echo "To push to Docker Hub (for Runpod):"
echo "  docker tag autointerp:latest <your-dockerhub-username>/autointerp:latest"
echo "  docker push <your-dockerhub-username>/autointerp:latest"
echo ""
echo "To use with Runpod:"
echo "  1. Push image to Docker Hub or container registry"
echo "  2. Update AUTOINTERP_CONTAINER_IMAGE in set_env_vars.sh"
echo "  3. Run: bash ../src/commit_utils/commit.sh"
