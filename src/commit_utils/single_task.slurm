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

# Variables injected via: sbatch --export=ALL,task=...,model=...,agent=...,agent_config=...,num_hours=...,num_gpus=...
bash src/run_task.sh \
    "${task}" "${agent}" "${model}" \
    "${SLURM_JOB_ID}" "${num_hours}" "${agent_config}" "${num_gpus:-1}"
