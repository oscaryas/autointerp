#!/bin/bash
set -e

# Parse arguments
TASK="$1"
AGENT="$2"
MODEL="$3"
AGENT_CONFIG="$4"
TIME_LIMIT="${5:-2}"

echo "=== AutoInterp Task Execution ==="
echo "Task: $TASK"
echo "Agent: $AGENT ($AGENT_CONFIG)"
echo "Model: $MODEL"
echo "Time Limit: ${TIME_LIMIT}h"
echo "=================================="

# Create working directory
WORK_DIR="/workspace/task_${TASK}_$$"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Copy task files
TASK_DIR="/workspace/AutoInterp/src/probing/tasks/${TASK}"
if [ ! -d "$TASK_DIR" ]; then
    echo "ERROR: Task directory not found: $TASK_DIR"
    exit 1
fi

echo "Copying task files..."
cp "$TASK_DIR/labeled_data.jsonl" .
cp "$TASK_DIR/evaluate.py" .
if [ -f "$TASK_DIR/task_description.txt" ]; then
    cp "$TASK_DIR/task_description.txt" .
fi
if [ -d "$TASK_DIR/task_context" ]; then
    cp -r "$TASK_DIR/task_context" .
fi

# Copy utility scripts
cp /workspace/AutoInterp/src/utils/*.py . 2>/dev/null || true

# Generate prompt
echo "Generating agent prompt..."
PROMPT=$(python /workspace/AutoInterp/src/probing/general/get_prompt.py \
    --task "$TASK" \
    --model "$MODEL" \
    --time-limit "$TIME_LIMIT" \
    --agent "$AGENT")

echo "$PROMPT" > prompt.txt
echo "Prompt saved to prompt.txt"

# Create timer script
cat > timer.sh <<'TIMER'
#!/bin/bash
LIMIT_HOURS=$1
START_TIME=$(date +%s)
CURRENT_TIME=$(date +%s)
ELAPSED_SECONDS=$((CURRENT_TIME - START_TIME))
LIMIT_SECONDS=$((LIMIT_HOURS * 3600))
REMAINING_SECONDS=$((LIMIT_SECONDS - ELAPSED_SECONDS))

if [ $REMAINING_SECONDS -le 0 ]; then
    echo "Time limit exceeded!"
    exit 1
fi

HOURS=$((REMAINING_SECONDS / 3600))
MINUTES=$(((REMAINING_SECONDS % 3600) / 60))
SECONDS=$((REMAINING_SECONDS % 60))
echo "Time remaining: ${HOURS}h ${MINUTES}m ${SECONDS}s"
TIMER
chmod +x timer.sh
echo "$TIME_LIMIT" > .time_limit

# Copy agent solve script
AGENT_SCRIPT="/workspace/AutoInterp/agents/${AGENT}/solve.sh"
if [ ! -f "$AGENT_SCRIPT" ]; then
    echo "ERROR: Agent script not found: $AGENT_SCRIPT"
    exit 1
fi
cp "$AGENT_SCRIPT" agent_solve.sh
chmod +x agent_solve.sh

# Set up logging
exec 1> >(tee output.log)
exec 2> >(tee error.log >&2)

echo "=== Starting Agent ==="
START_TIME=$(date +%s)

# Run agent with timeout
timeout "${TIME_LIMIT}h" bash agent_solve.sh || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "Agent timed out after ${TIME_LIMIT}h"
    else
        echo "Agent exited with code $EXIT_CODE"
    fi
}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Agent Finished ==="
echo "Elapsed time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"

# Check if final_probe exists
if [ -d "final_probe" ]; then
    echo "Found final_probe directory"
    ls -la final_probe/
else
    echo "WARNING: No final_probe directory found"
    mkdir -p final_probe
fi

# Run evaluation
echo "=== Running Evaluation ==="
if python evaluate.py --model-path "$MODEL" --probe-path final_probe --json-output-file metrics.json; then
    echo "Evaluation completed successfully"
    cat metrics.json
else
    echo "ERROR: Evaluation failed"
    echo '{"accuracy": 0.0, "error": "evaluation_failed"}' > metrics.json
fi

# Save results
RESULT_DIR="/workspace/results/${AGENT}_${AGENT_CONFIG}/${TASK}_${MODEL//\//__}_$(date +%s)"
mkdir -p "$RESULT_DIR"

echo "=== Saving Results ==="
cp output.log "$RESULT_DIR/"
cp error.log "$RESULT_DIR/"
cp prompt.txt "$RESULT_DIR/"
cp metrics.json "$RESULT_DIR/"
cp -r final_probe "$RESULT_DIR/" 2>/dev/null || true

# Copy any generated analysis files
cp *.png "$RESULT_DIR/" 2>/dev/null || true
cp *.pdf "$RESULT_DIR/" 2>/dev/null || true
cp *.html "$RESULT_DIR/" 2>/dev/null || true

echo "Results saved to: $RESULT_DIR"
echo "=== Task Complete ==="

# Cleanup
cd /workspace
rm -rf "$WORK_DIR"
