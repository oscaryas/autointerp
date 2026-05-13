#!/bin/bash
set -e

# Load environment variables
source "$(dirname "${BASH_SOURCE[0]}")/set_env_vars.sh"

# Models to probe
models=(
    "google/gemma-2-2b"
    "Qwen/Qwen2.5-1.5B"
    "HuggingFaceTB/SmolLM2-1.7B"
)

# Probing tasks
tasks=(
    "sentiment_analysis"
)

# Agents to use
agents=(
    "claude:claude-opus-4-6"
    "claude:claude-sonnet-4-5"
    "codex:gpt-4"
)

echo "=== AutoInterp Job Submission ==="
echo "Tasks: ${tasks[@]}"
echo "Models: ${models[@]}"
echo "Agents: ${agents[@]}"
echo "================================="
echo ""

# Loop over all combinations
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for agent_spec in "${agents[@]}"; do
            # Parse agent:config
            IFS=':' read -r agent agent_config <<< "$agent_spec"

            echo "Submitting: $agent ($agent_config) on $task with $model"

            # Submit job
            bash "$(dirname "${BASH_SOURCE[0]}")/single_task.sh" \
                --task "$task" \
                --model "$model" \
                --agent "$agent" \
                --agent-config "$agent_config" \
                --time-limit "${AUTOINTERP_TIME_LIMIT}"

            # Sleep to avoid API rate limits
            sleep 5
        done
    done
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor jobs with: runpodctl get pods"
echo "View logs with: runpodctl logs <pod-id> --follow"
echo "Results will be saved to: ${AUTOINTERP_RESULTS_DIR}"
