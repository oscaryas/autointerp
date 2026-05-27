# HPC Migration Plan: Runpod -> CSCS Clariden

AutoInterp currently submits jobs to Runpod with `runpodctl` and runs `src/run_task.sh` inside an ephemeral Docker container. On CSCS Clariden, the target architecture should be **Slurm + CSCS Container Engine**, not PostTrainBench's HTCondor + Apptainer setup.

This plan is AutoInterp-specific. PostTrainBench is useful as a reference for the benchmark lifecycle, but its scheduler/container details should not be copied directly. AutoInterp should keep its probing task layout:

- task inputs under `src/probing/tasks/<task>/`
- utility scripts under `src/utils/`
- agent launchers under `agents/<agent>/solve.sh`
- final artifact `final_probe/`
- evaluator called as `evaluate.py --model-path ... --probe-path final_probe --json-output-file ...`

## Clariden Model

Clariden uses Slurm. Jobs should be submitted with `sbatch`, monitored with `squeue`, and charged to a CSCS project account via `--account`.

Clariden's recommended ML container path is CSCS Container Engine. Container Engine uses EDF TOML files and Slurm's `--environment` option. The basic launch shape is:

```bash
srun --environment=/path/to/autointerp.toml bash -lc '...'
```

Use a scratch/project filesystem for the repository, model cache, containers, job workspaces, and results. Do not rely on `$HOME` for large files. Prefer placing the repository under `AUTOINTERP_SCRATCH` or another project/scratch path that can be mounted into Container Engine jobs.

## Important Differences From PostTrainBench

- Do not create `src/commit_utils/single_task.sub`; Clariden needs `single_task.slurm`.
- Do not use `condor_submit_bid`; use `sbatch`.
- Do not use Apptainer `.sif` as the primary runtime target; use CSCS Container Engine with an EDF.
- Do not use `fuse-overlayfs`; use pre-populated HuggingFace cache directories on scratch.
- Do not copy PostTrainBench's `job_dir/task` layout unless all paths are changed. AutoInterp can run directly from a per-job workspace containing `evaluate.py`, `labeled_data.jsonl`, utility scripts, and `agent_solve.sh`.

## Preconditions

- A CSCS account/project allocation is available. Set it as `CSCS_ACCOUNT`.
- The AutoInterp repository is on a shared Clariden-visible filesystem, preferably scratch or project storage.
- The selected agent API keys are exported before submission: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and/or `OPENCODE_API_KEY`.
- HuggingFace models are pre-populated in a shared cache before submitting the job matrix.
- Container image choice is compatible with Clariden GH200/aarch64 nodes. Prefer CSCS/NGC/Alps-compatible PyTorch images over generic x86 CUDA images.
- Fix the current probe evaluation scaler issue before trusting benchmark metrics. See "Required AutoInterp Fix" below.

## Files To Create

### `containers/Containerfile`

Create an AutoInterp container recipe for Podman. This replaces the current Docker-only workflow for Clariden.

Recommended base:

- Start from an aarch64-compatible NGC/CSCS PyTorch image where possible.
- If using a CUDA Ubuntu base, verify that the image supports Clariden's architecture before building.

Install only the AutoInterp runtime surface:

- Python: `torch`, `transformers`, `accelerate`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `baukit`, `nnsight`, `matplotlib`, `seaborn`, `plotly`, `datasets`, `evaluate`, `rouge-score`, `safetensors`, `sentencepiece`, `protobuf`, `tokenizers`, `openai`, `anthropic`, `jsonlines`, `pyyaml`, `tqdm`, `huggingface_hub`
- CLI agents: `@anthropic-ai/claude-code@2.0.55`, `@openai/codex@0.79.0`, `opencode-ai@1.1.59`
- Utilities: `git`, `curl`, `wget`, `tree`, `vim`, `tmux`, `htop`

Use `uv pip install --system` if `uv` is installed in the image; otherwise install `uv` first.

### `containers/autointerp.toml.template`

Create a Container Engine EDF template. Keep host and container paths identical by mounting the scratch/project root to itself. This simplifies `run_task.sh` because paths generated on the host still exist inside the container.

Mount `AUTOINTERP_ROOT` separately only if the repository is outside `AUTOINTERP_SCRATCH`. If the repository is inside `AUTOINTERP_SCRATCH`, the `AUTOINTERP_ROOT` mount is redundant and can be omitted.

```toml
image = "${AUTOINTERP_CONTAINER_IMAGE}"
writable = true
mounts = [
  "${AUTOINTERP_ROOT}:${AUTOINTERP_ROOT}",
  "${AUTOINTERP_SCRATCH}:${AUTOINTERP_SCRATCH}",
  "${AUTOINTERP_HF_HOME}:${AUTOINTERP_HF_HOME}"
]
workdir = "${AUTOINTERP_ROOT}"

[env]
HF_HOME = "${AUTOINTERP_HF_HOME}"
HF_HUB_DISABLE_TELEMETRY = "1"
PYTHONUNBUFFERED = "1"
PYTHONDONTWRITEBYTECODE = "1"
NO_PROXY = "localhost,127.0.0.1"
no_proxy = "localhost,127.0.0.1"
```

Generate a concrete EDF at setup time, for example `${AUTOINTERP_SCRATCH}/containers/autointerp.toml`, after the image path and scratch paths are known.

Minimal generation command:

```bash
mkdir -p "${AUTOINTERP_CONTAINERS_DIR}"
envsubst < containers/autointerp.toml.template > "${AUTOINTERP_EDF}"
```

If `envsubst` is unavailable, generate this file with a short Python script or manually substitute the paths.

`AUTOINTERP_CONTAINER_IMAGE` can be:

- a local `.sqsh` image path produced by `enroot import`, or
- a remote OCI image reference supported by CSCS Container Engine.

For repeatable benchmark runs, prefer a local `.sqsh` image on scratch.

### `src/commit_utils/single_task.slurm`

Create a Slurm batch script. Values are supplied by `sbatch --export=...` from `commit.sh`.

```bash
#!/bin/bash
#SBATCH --job-name=autointerp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

bash src/run_task.sh \
    "${task}" "${agent}" "${model}" \
    "${SLURM_JOB_ID}" "${num_hours}" "${agent_config}" "${num_gpus:-1}"
```

Verify the correct GPU request syntax on Clariden before implementation. If `#SBATCH --gpus=1` is not the site-preferred form, replace it with the CSCS-recommended directive, such as `--gpus-per-node=1` or another partition-specific option. Check with:

```bash
sinfo -o "%P %N %G"
scontrol show partition
```

Clariden normal/debug nodes are generally not shared, so a one-GPU AutoInterp job may still reserve a full node. Treat that as an allocation-cost issue when scaling the matrix.

## Files To Modify

### `containers/build_container.sh`

Replace Docker/Runpod build instructions with a Clariden-compatible build/import flow.

```bash
#!/bin/bash
set -euo pipefail

container="${1:-autointerp}"

: "${AUTOINTERP_CONTAINER_TAG:=autointerp:${container}}"
: "${AUTOINTERP_CONTAINERS_DIR:=${SCRATCH:-$PWD}/autointerp/containers}"

mkdir -p "${AUTOINTERP_CONTAINERS_DIR}"

podman build -t "${AUTOINTERP_CONTAINER_TAG}" -f containers/Containerfile .

enroot import \
    -o "${AUTOINTERP_CONTAINERS_DIR}/${container}.sqsh" \
    "podman://${AUTOINTERP_CONTAINER_TAG}"

echo "Built ${AUTOINTERP_CONTAINERS_DIR}/${container}.sqsh"
```

If building on Clariden is inconvenient, build and push a multi-arch OCI image elsewhere, then reference it from the EDF or import it with `enroot`.

### `src/commit_utils/set_env_vars.sh`

Remove Runpod variables:

- `RUNPOD_API_KEY`
- `RUNPOD_GPU_TYPE`
- `RUNPOD_GPU_COUNT`
- `RUNPOD_DISK_SIZE`
- `RUNPOD_CLOUD_TYPE`
- `RUNPOD_REGION`
- `AUTOINTERP_CONTAINER_IMAGE` as a Docker image for Runpod

Add Clariden variables:

```bash
set_default AUTOINTERP_JOB_SCHEDULER "slurm_clariden"
set_default CSCS_ACCOUNT ""

set_default AUTOINTERP_ROOT "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set_default AUTOINTERP_SCRATCH "${SCRATCH:-/capstor/scratch/cscs/${USER}/autointerp}"
set_default AUTOINTERP_RESULTS_DIR "${AUTOINTERP_SCRATCH}/results"
set_default AUTOINTERP_CONTAINERS_DIR "${AUTOINTERP_SCRATCH}/containers"
set_default AUTOINTERP_CONTAINER_NAME "autointerp"
set_default AUTOINTERP_CONTAINER_IMAGE "${AUTOINTERP_CONTAINERS_DIR}/${AUTOINTERP_CONTAINER_NAME}.sqsh"
set_default AUTOINTERP_EDF "${AUTOINTERP_CONTAINERS_DIR}/autointerp.toml"
set_default AUTOINTERP_HF_HOME "${AUTOINTERP_SCRATCH}/hf_cache"
set_default AUTOINTERP_TIME_LIMIT "2"
set_default AUTOINTERP_EXPERIMENT_NAME ""

set_default HF_HOME "${AUTOINTERP_HF_HOME}"
set_default HF_TOKEN "${HF_TOKEN:-}"
```

Validation should warn if:

- `CSCS_ACCOUNT` is empty
- `${AUTOINTERP_EDF}` does not exist
- `${AUTOINTERP_CONTAINER_IMAGE}` does not exist when using a local `.sqsh`
- no agent API key is set

### `src/commit_utils/commit.sh`

Replace the Runpod loop with Slurm submissions. Keep AutoInterp's local model/task/agent arrays.

```bash
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
)

agents=(
    "claude:claude-opus-4-6"
    "codex:gpt-4"
)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for agent_spec in "${agents[@]}"; do
            IFS=':' read -r agent agent_config <<< "$agent_spec"

            sbatch \
                --account="${CSCS_ACCOUNT}" \
                --time="$((AUTOINTERP_TIME_LIMIT + 1)):00:00" \
                --export=ALL,task="${task}",model="${model}",agent="${agent}",agent_config="${agent_config}",num_hours="${AUTOINTERP_TIME_LIMIT}",num_gpus=1 \
                src/commit_utils/single_task.slurm

            sleep 5
        done
    done
done
```

The Slurm time limit adds one hour of padding so the agent timeout can fire and evaluation can still run.

### `src/run_task.sh`

Rewrite `run_task.sh` so it runs on the Slurm worker node and launches AutoInterp agent/evaluation commands inside CSCS Container Engine.

Argument order:

```bash
TASK="$1"
AGENT="$2"
MODEL="$3"
CLUSTER_ID="$4"      # SLURM_JOB_ID
NUM_HOURS="$5"
AGENT_CONFIG="$6"
NUM_GPUS="${7:-1}"
```

Structure:

1. Source `src/commit_utils/set_env_vars.sh`.

2. Create result directory:

   ```bash
   MODEL_SAFE=$(echo "$MODEL" | tr '/:[]' '____')
   AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:[]' '____')
   EVAL_DIR="${AUTOINTERP_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${AUTOINTERP_EXPERIMENT_NAME}/${TASK}_${MODEL_SAFE}_${CLUSTER_ID}"
   mkdir -p "$EVAL_DIR"
   exec 1>"${EVAL_DIR}/output.log"
   exec 2>"${EVAL_DIR}/error.log"
   ```

3. Create an AutoInterp job workspace on scratch, not `/tmp`:

   ```bash
   JOB_DIR="${AUTOINTERP_SCRATCH}/jobs/${TASK}_${MODEL_SAFE}_${CLUSTER_ID}"
   JOB_TMP="${JOB_DIR}/tmp"
   mkdir -p "$JOB_DIR" "$JOB_TMP"
   ```

4. Copy AutoInterp task files into `JOB_DIR`:

   ```bash
   TASK_DIR="${AUTOINTERP_ROOT}/src/probing/tasks/${TASK}"
   cp "${TASK_DIR}/labeled_data.jsonl" "$JOB_DIR/"
   cp "${TASK_DIR}/evaluate.py" "$JOB_DIR/"
   [ -f "${TASK_DIR}/task_description.txt" ] && cp "${TASK_DIR}/task_description.txt" "$JOB_DIR/"
   [ -d "${TASK_DIR}/task_context" ] && cp -r "${TASK_DIR}/task_context" "$JOB_DIR/"
   cp "${AUTOINTERP_ROOT}/src/utils/"*.py "$JOB_DIR/"
   cp "${AUTOINTERP_ROOT}/agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"
   chmod +x "${JOB_DIR}/agent_solve.sh"
   ```

5. Generate the AutoInterp prompt:

   ```bash
   PROMPT=$(python "${AUTOINTERP_ROOT}/src/probing/general/get_prompt.py" \
       --task "$TASK" \
       --model "$MODEL" \
       --time-limit "$NUM_HOURS" \
       --agent "$AGENT")
   echo "$PROMPT" > "${JOB_DIR}/prompt.txt"
   cp "${JOB_DIR}/prompt.txt" "${EVAL_DIR}/prompt.txt"
   ```

6. Create `timer.sh` in `JOB_DIR`. The AutoInterp prompt tells agents to run `bash timer.sh`, so the file must exist in the container working directory.

7. Export agent environment:

   ```bash
   export CODEX_API_KEY="${CODEX_API_KEY:-${OPENAI_API_KEY:-}}"
   export PROMPT
   export AGENT_CONFIG
   export HF_HOME="${AUTOINTERP_HF_HOME}"
   ```

   Make API key propagation explicit. The `sbatch --export=ALL,...` call passes host variables into the Slurm job, and `srun --environment="${AUTOINTERP_EDF}"` must preserve the variables needed by agent CLIs. Before scaling, verify this with a tiny test command inside the Container Engine environment:

   ```bash
   srun --environment="${AUTOINTERP_EDF}" bash -c 'python - <<PY
   import os
   for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY", "GEMINI_API_KEY", "OPENCODE_API_KEY", "HF_HOME"]:
       print(key, "set" if os.environ.get(key) else "missing")
   PY'
   ```

   If any required key is missing, add it to the generated EDF `[env]` section or pass it through using the CSCS-recommended Container Engine environment mechanism. Do not write secret values into a committed template; generate the concrete EDF on the cluster from environment variables or another private setup step.

8. Run the agent inside the Container Engine environment:

   ```bash
   solve_task() {
       timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
       srun --environment="${AUTOINTERP_EDF}" \
           --ntasks=1 \
           --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
           bash -c "cd '${JOB_DIR}' && bash ./agent_solve.sh 2>&1 | python ./timestamp_lines.py" \
           > "${EVAL_DIR}/solve_out.txt" 2>&1
   }
   ```

9. Parse the agent trace if a parser exists:

   ```bash
   TRACE_PARSER="${AUTOINTERP_ROOT}/agents/${AGENT}/human_readable_trace.py"
   if [ -f "$TRACE_PARSER" ]; then
       python "$TRACE_PARSER" "${EVAL_DIR}/solve_out.txt" -o "${EVAL_DIR}/solve_parsed.txt"
   fi
   ```

10. Ensure `final_probe/` exists:

    ```bash
    [ -d "${JOB_DIR}/final_probe" ] || mkdir -p "${JOB_DIR}/final_probe"
    ```

11. Run the evaluator inside the same Container Engine environment:

    ```bash
    srun --environment="${AUTOINTERP_EDF}" \
        --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
        bash -c "cd '${JOB_DIR}' && python evaluate.py \
            --model-path '${MODEL}' \
            --probe-path final_probe \
            --json-output-file '${EVAL_DIR}/metrics.json'"
    ```

12. Copy artifacts:

    ```bash
    cp -r "${JOB_DIR}/final_probe" "${EVAL_DIR}/" 2>/dev/null || true
    cp "${JOB_DIR}"/analysis.* "${EVAL_DIR}/" 2>/dev/null || true
    cp "${JOB_DIR}"/*.png "${EVAL_DIR}/" 2>/dev/null || true
    cp "${JOB_DIR}"/*.pdf "${EVAL_DIR}/" 2>/dev/null || true
    cp "${JOB_DIR}"/*.html "${EVAL_DIR}/" 2>/dev/null || true
    ```

13. Cleanup job workspace only after artifacts are copied.

## Files To Delete Or Stop Using

- `src/commit_utils/single_task.sh`: Runpod-specific.
- Runpod-specific documentation snippets in `README.md` and `CLAUDE.md`, once the Slurm path is implemented.

Keep `containers/Dockerfile` only if you still want local Docker development. Clariden should use `containers/Containerfile` + EDF.

## Required AutoInterp Fix

Fix `src/probing/tasks/sentiment_analysis/evaluate.py` so saved scalers are applied before prediction:

```python
probe = probe_data.get("model") or probe_data.get("probe") or probe_data
scaler = probe_data.get("scaler") if isinstance(probe_data, dict) else None
if scaler is not None:
    X = scaler.transform(X)
y_pred = probe.predict(X)
```

Without this, probes trained with `src/utils/train_probe.py` are evaluated on unnormalized activations, producing invalid metrics.

Also consider fixing padding-aware pooling in `evaluate.py` and `src/utils/extract_activations.py`; current mean pooling includes padded tokens.

## Setup And Verification

1. Put the repo on scratch/project storage. On Clariden `$SCRATCH` = `/iopsstor/scratch/cscs/$USER` (NVMe, 14-day cleanup) — prefer this over `/capstor` for active job workspaces:

   ```bash
   cd $SCRATCH
   git clone <repo> autointerp
   cd autointerp
   ```

2. Set environment:

   ```bash
   export CSCS_ACCOUNT=<your-project-account>
   export AUTOINTERP_SCRATCH=${SCRATCH}/autointerp_runs
   export AUTOINTERP_RESULTS_DIR=${AUTOINTERP_SCRATCH}/results
   export AUTOINTERP_CONTAINERS_DIR=${AUTOINTERP_SCRATCH}/containers
   export AUTOINTERP_HF_HOME=${AUTOINTERP_SCRATCH}/hf_cache
   export AUTOINTERP_CONTAINER_IMAGE=${AUTOINTERP_CONTAINERS_DIR}/autointerp.sqsh
   export AUTOINTERP_EDF=${AUTOINTERP_CONTAINERS_DIR}/autointerp.toml
   export AUTOINTERP_JOB_SCHEDULER=slurm_clariden
   export AUTOINTERP_TIME_LIMIT=2
   ```

3. Build/import the container or point `AUTOINTERP_CONTAINER_IMAGE` at an existing compatible image.

4. Generate the concrete EDF from `containers/autointerp.toml.template`, substituting the exported paths.

   ```bash
   mkdir -p "${AUTOINTERP_CONTAINERS_DIR}"
   envsubst < containers/autointerp.toml.template > "${AUTOINTERP_EDF}"
   ```

5. Pre-populate HuggingFace cache:

   ```bash
   export HF_HOME="${AUTOINTERP_HF_HOME}"
   mkdir -p "${HF_HOME}"

   python - <<'PY'
   from huggingface_hub import snapshot_download

   for repo in [
       "google/gemma-2-2b",
       "Qwen/Qwen2.5-1.5B",
       "HuggingFaceTB/SmolLM2-1.7B",
   ]:
       snapshot_download(repo)
   PY
   ```

6. Submit one test job:

   ```bash
   export ANTHROPIC_API_KEY=...

   sbatch \
     --account="${CSCS_ACCOUNT}" \
     --time=03:00:00 \
     --export=ALL,task=sentiment_analysis,model=google/gemma-2-2b,agent=claude,agent_config=claude-opus-4-6,num_hours=2,num_gpus=1 \
     src/commit_utils/single_task.slurm
   ```

7. Monitor and inspect:

   ```bash
   squeue -u "$USER"
   tail -f "${AUTOINTERP_RESULTS_DIR}"/claude_claude-opus-4-6_2h/*/output.log
   ```

8. Before submitting the full matrix, inspect:

   - `output.log`
   - `error.log`
   - `solve_out.txt`
   - `solve_parsed.txt` if present
   - `metrics.json`
