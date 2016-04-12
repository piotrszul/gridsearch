#!/bin/bash

#SBATCH --job-name="search"
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=131072

module load python/2.7.9
$HOME/.local/bin/fab run_task:${SLURM_ARRAY_TASK_ID:-0}
