#!/bin/sh
#SBATCH --job-name=shglncp
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --mail-user=sl4742@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-161
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/shallow_gln_cp.py $SLURM_ARRAY_TASK_ID
