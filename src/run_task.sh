#!/bin/bash

TASK="$1"
AGENT="$2"
MODEL="$3"
CLUSTER_ID="$4"      # SLURM_JOB_ID passed from single_task.slurm
NUM_HOURS="$5"
AGENT_CONFIG="$6"
NUM_GPUS="${7:-1}"

source "$(dirname "${BASH_SOURCE[0]}")/commit_utils/set_env_vars.sh"

MODEL_SAFE=$(echo "$MODEL" | tr '/:[]' '____')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:[]' '____')
export TASK MODEL

EVAL_DIR="${AUTOINTERP_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${AUTOINTERP_EXPERIMENT_NAME}/${TASK}_${MODEL_SAFE}_${CLUSTER_ID}"
mkdir -p "$EVAL_DIR"

exec 1>"${EVAL_DIR}/output.log"
exec 2>"${EVAL_DIR}/error.log"

echo "$@"

# Job workspace on scratch (not /tmp — must be visible to Container Engine mounts)
JOB_DIR="${AUTOINTERP_SCRATCH}/jobs/${TASK}_${MODEL_SAFE}_${CLUSTER_ID}"
JOB_TMP="${JOB_DIR}/tmp"
mkdir -p "$JOB_DIR" "$JOB_TMP"

echo "Preparing job directory..."

# Copy task files
TASK_DIR="${AUTOINTERP_ROOT}/src/probing/tasks/${TASK}"
[ -f "${TASK_DIR}/labeled_data.jsonl" ] && cp "${TASK_DIR}/labeled_data.jsonl" "$JOB_DIR/"
cp "${TASK_DIR}/evaluate.py" "$JOB_DIR/"
[ -f "${TASK_DIR}/task_description.txt" ] && cp "${TASK_DIR}/task_description.txt" "$JOB_DIR/"
[ -d "${TASK_DIR}/task_context" ] && cp -r "${TASK_DIR}/task_context" "$JOB_DIR/"

# Copy utility scripts (agents import these directly from their working dir)
cp "${AUTOINTERP_ROOT}/src/utils/"*.py "$JOB_DIR/"

# Copy agent launcher
cp "${AUTOINTERP_ROOT}/agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"
chmod +x "${JOB_DIR}/agent_solve.sh"

# Generate prompt
PROMPT=$(python3 "${AUTOINTERP_ROOT}/src/probing/general/get_prompt.py" \
    --task "$TASK" \
    --model "$MODEL" \
    --time-limit "$NUM_HOURS" \
    --agent "$AGENT")
echo "$PROMPT" > "${JOB_DIR}/prompt.txt"
cp "${JOB_DIR}/prompt.txt" "${EVAL_DIR}/prompt.txt"

# Create timer.sh with start time baked in (agents call `bash timer.sh` to check remaining time)
cat > "${JOB_DIR}/timer.sh" <<EOF
#!/bin/bash
LIMIT_SECONDS=$(( NUM_HOURS * 3600 ))
START_EPOCH=$(date +%s)
CURRENT=\$(date +%s)
REMAINING=\$(( LIMIT_SECONDS - (CURRENT - START_EPOCH) ))
if [ \$REMAINING -le 0 ]; then
    echo "Time limit exceeded!"
    exit 1
fi
printf 'Time remaining: %dh %dm %ds\n' \$(( REMAINING / 3600 )) \$(( (REMAINING % 3600) / 60 )) \$(( REMAINING % 60 ))
EOF
chmod +x "${JOB_DIR}/timer.sh"

# Propagate agent environment variables
export CODEX_API_KEY="${CODEX_API_KEY:-${OPENAI_API_KEY:-}}"
export PROMPT
export AGENT_CONFIG
export HF_HOME="${AUTOINTERP_HF_HOME}"

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"
echo "Task:     $TASK"
echo "Agent:    $AGENT ($AGENT_CONFIG)"
echo "Model:    $MODEL"
echo "Job dir:  $JOB_DIR"
echo "Eval dir: $EVAL_DIR"

with_record_the_time() {
    local begin
    begin=$(date --iso-8601=seconds)
    "$@"
    local exit_code=$?
    local end
    end=$(date --iso-8601=seconds)
    local time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
    printf '%02d:%02d:%02d\n' \
        $(( time_taken / 3600 )) \
        $(( (time_taken % 3600) / 60 )) \
        $(( time_taken % 60 )) > "${EVAL_DIR}/time_taken.txt"
    return $exit_code
}

solve_task() {
    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    srun --environment="${AUTOINTERP_EDF}" \
        --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        bash -c "cd '${JOB_DIR}' && bash ./agent_solve.sh 2>&1 | python ./timestamp_lines.py" \
        > "${EVAL_DIR}/solve_out.txt" 2>&1
}

with_record_the_time solve_task
SOLVE_EXIT=$?

echo "--- SOLVE DIAGNOSTICS ---"
echo "exit_code: $SOLVE_EXIT"
if [ $SOLVE_EXIT -eq 0 ]; then
    echo "status: exited normally"
elif [ $SOLVE_EXIT -eq 124 ]; then
    echo "status: killed by timeout (reached ${NUM_HOURS}h limit)"
elif [ $SOLVE_EXIT -gt 128 ]; then
    echo "status: killed by signal $((SOLVE_EXIT - 128))"
else
    echo "status: exited with error code $SOLVE_EXIT"
fi
echo "hostname: $(hostname)"
echo "disk_job_dir: $(du -sh "${JOB_DIR}" 2>/dev/null | cut -f1)"
echo "--- END SOLVE DIAGNOSTICS ---"

# Parse agent trace
TRACE_PARSER="${AUTOINTERP_ROOT}/agents/${AGENT}/human_readable_trace.py"
if [ -f "$TRACE_PARSER" ]; then
    python3 "$TRACE_PARSER" "${EVAL_DIR}/solve_out.txt" -o "${EVAL_DIR}/solve_parsed.txt" || true
fi

# Ensure final_probe exists even if agent didn't produce one
[ -d "${JOB_DIR}/final_probe" ] || mkdir -p "${JOB_DIR}/final_probe"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

srun --environment="${AUTOINTERP_EDF}" \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    bash -c "cd '${JOB_DIR}' && python evaluate.py \
        --model-path '${MODEL}' \
        --probe-path final_probe \
        --json-output-file '${EVAL_DIR}/metrics.json'" || true

echo "================================"
echo "===== COLLECTING ARTIFACTS ====="
echo "================================"

cp -r "${JOB_DIR}/final_probe" "${EVAL_DIR}/" 2>/dev/null || true
cp "${JOB_DIR}"/analysis.* "${EVAL_DIR}/" 2>/dev/null || true
cp "${JOB_DIR}"/*.png "${EVAL_DIR}/" 2>/dev/null || true
cp "${JOB_DIR}"/*.pdf "${EVAL_DIR}/" 2>/dev/null || true
cp "${JOB_DIR}"/*.html "${EVAL_DIR}/" 2>/dev/null || true
cp "${JOB_DIR}"/*.ipynb "${EVAL_DIR}/" 2>/dev/null || true

rm -rf "$JOB_DIR"

echo "=== Task Complete ==="
echo "Results: $EVAL_DIR"
