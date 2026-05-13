#!/bin/bash
set -e

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --agent-config)
            AGENT_CONFIG="$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$TASK" ] || [ -z "$MODEL" ] || [ -z "$AGENT" ] || [ -z "$AGENT_CONFIG" ]; then
    echo "Usage: $0 --task <task> --model <model> --agent <agent> --agent-config <config> [--time-limit <hours>]"
    exit 1
fi

# Load environment variables
source "$(dirname "${BASH_SOURCE[0]}")/set_env_vars.sh"

# Set defaults
TIME_LIMIT="${TIME_LIMIT:-2}"

# Create safe names for paths
MODEL_SAFE=$(echo "$MODEL" | tr '/:' '__')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '__')
RANDOM_ID=$(date +%s)_${RANDOM}

# Result directory
RESULT_DIR="${AUTOINTERP_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}${AUTOINTERP_EXPERIMENT_NAME}/${TASK}_${MODEL_SAFE}_${RANDOM_ID}"
mkdir -p "$RESULT_DIR"

echo "=== Submitting Runpod Job ==="
echo "Task: $TASK"
echo "Model: $MODEL"
echo "Agent: $AGENT ($AGENT_CONFIG)"
echo "Time Limit: ${TIME_LIMIT}h"
echo "Result Dir: $RESULT_DIR"
echo "============================="

# Create job script that will run on Runpod
JOB_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -e

# Environment
export TASK="__TASK__"
export MODEL="__MODEL__"
export AGENT="__AGENT__"
export AGENT_CONFIG="__AGENT_CONFIG__"
export TIME_LIMIT="__TIME_LIMIT__"
export ANTHROPIC_API_KEY="__ANTHROPIC_API_KEY__"
export OPENAI_API_KEY="__OPENAI_API_KEY__"
export GEMINI_API_KEY="__GEMINI_API_KEY__"
export HF_TOKEN="__HF_TOKEN__"

# Clone repository (or copy if mounted)
if [ ! -d "/workspace/AutoInterp" ]; then
    echo "Setting up workspace..."
    # In a real scenario, you'd clone your repo or have it pre-baked in the container
    # For now, assume it's mounted or copied
fi

cd /workspace

# Run the task
bash AutoInterp/src/run_task.sh "$TASK" "$AGENT" "$MODEL" "$AGENT_CONFIG" "$TIME_LIMIT"

# Copy results to output location
cp -r /workspace/results/* /workspace/output/
EOF
)

# Replace placeholders
JOB_SCRIPT="${JOB_SCRIPT//__TASK__/$TASK}"
JOB_SCRIPT="${JOB_SCRIPT//__MODEL__/$MODEL}"
JOB_SCRIPT="${JOB_SCRIPT//__AGENT__/$AGENT}"
JOB_SCRIPT="${JOB_SCRIPT//__AGENT_CONFIG__/$AGENT_CONFIG}"
JOB_SCRIPT="${JOB_SCRIPT//__TIME_LIMIT__/$TIME_LIMIT}"
JOB_SCRIPT="${JOB_SCRIPT//__ANTHROPIC_API_KEY__/${ANTHROPIC_API_KEY:-}}"
JOB_SCRIPT="${JOB_SCRIPT//__OPENAI_API_KEY__/${OPENAI_API_KEY:-}}"
JOB_SCRIPT="${JOB_SCRIPT//__GEMINI_API_KEY__/${GEMINI_API_KEY:-}}"
JOB_SCRIPT="${JOB_SCRIPT//__HF_TOKEN__/${HF_TOKEN:-}}"

# Save job script
JOB_SCRIPT_FILE="${RESULT_DIR}/job_script.sh"
echo "$JOB_SCRIPT" > "$JOB_SCRIPT_FILE"
chmod +x "$JOB_SCRIPT_FILE"

# Submit to Runpod
# Note: This is a simplified example. In practice, you'd use runpodctl or the Runpod API
echo "Submitting to Runpod..."

if command -v runpodctl &> /dev/null; then
    # Using runpodctl (if available)
    POD_ID=$(runpodctl create pod \
        --name "autointerp-${TASK}-${RANDOM_ID}" \
        --gpuType "${RUNPOD_GPU_TYPE}" \
        --gpuCount "${RUNPOD_GPU_COUNT}" \
        --imageName "${AUTOINTERP_CONTAINER_IMAGE}" \
        --containerDiskSize "${RUNPOD_DISK_SIZE}" \
        --volumeSize 50 \
        --env "TASK=${TASK}" \
        --env "MODEL=${MODEL}" \
        --env "AGENT=${AGENT}" \
        --env "AGENT_CONFIG=${AGENT_CONFIG}" \
        --env "TIME_LIMIT=${TIME_LIMIT}" \
        --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
        --env "OPENAI_API_KEY=${OPENAI_API_KEY:-}" \
        --env "GEMINI_API_KEY=${GEMINI_API_KEY:-}" \
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        --command "bash /workspace/job_script.sh" \
        --cloudType "${RUNPOD_CLOUD_TYPE}" \
        2>&1 | grep -oP 'Pod ID: \K[a-z0-9-]+' | head -1)

    echo "Pod created: $POD_ID"
    echo "$POD_ID" > "${RESULT_DIR}/pod_id.txt"

    # Save job info
    cat > "${RESULT_DIR}/job_info.json" <<JSON
{
  "pod_id": "$POD_ID",
  "task": "$TASK",
  "model": "$MODEL",
  "agent": "$AGENT",
  "agent_config": "$AGENT_CONFIG",
  "time_limit": "$TIME_LIMIT",
  "submitted_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
JSON

    echo "Job submitted successfully!"
    echo "Monitor with: runpodctl logs $POD_ID --follow"
else
    echo "ERROR: runpodctl not found. Install it or use Runpod API directly."
    echo ""
    echo "To install runpodctl:"
    echo "  wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -O runpodctl"
    echo "  chmod +x runpodctl"
    echo "  sudo mv runpodctl /usr/local/bin/"
    echo ""
    echo "Alternatively, you can call the Runpod API directly:"
    echo "  See: https://docs.runpod.io/reference/create-pod"
    exit 1
fi
