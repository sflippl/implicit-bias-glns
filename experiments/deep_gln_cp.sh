#!/bin/sh
#SBATCH --job-name=dpglncp
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00:00
#SBATCH --mail-user=sl4742@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-287
#SBATCH --output=slurm/slurm-%A_%a.out

python experiments/deep_gln_cp.py $SLURM_ARRAY_TASK_ID
