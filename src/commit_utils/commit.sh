#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/set_env_vars.sh"

models=(
    "google/gemma-2-2b"
    "Qwen/Qwen2.5-1.5B"
    "HuggingFaceTB/SmolLM2-1.7B"
)

tasks=(
    "sentiment_analysis"
    "deception"
    "refusal"
    "truth_analysis"
)

agents=(
    "claude:claude-opus-4-6"
    "codex:gpt-4"
)

echo "=== AutoInterp Job Submission ==="
echo "Tasks:  ${tasks[*]}"
echo "Models: ${models[*]}"
echo "Agents: ${agents[*]}"
echo "================================="

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for agent_spec in "${agents[@]}"; do
            IFS=':' read -r agent agent_config <<< "$agent_spec"

            echo "Submitting: $agent ($agent_config) | $task | $model"

            sbatch \
                --account="${CSCS_ACCOUNT}" \
                --time="$((AUTOINTERP_TIME_LIMIT + 1)):00:00" \
                --export=ALL,task="${task}",model="${model}",agent="${agent}",agent_config="${agent_config}",num_hours="${AUTOINTERP_TIME_LIMIT}",num_gpus=1 \
                src/commit_utils/single_task.slurm

            sleep 5
        done
    done
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ${AUTOINTERP_RESULTS_DIR}/*/*/output.log"
