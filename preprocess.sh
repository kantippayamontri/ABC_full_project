#!/usr/bin/bash
#SBATCH --job-name=ABC_pre		# create short name for job
#SBATCH --nodes=1			# node count
#SBATCH --ntasks=1			# total number of task across all nodes
#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=02:00:00			# total run time limit(HH:MM:SS)

module purge
module load anaconda3
conda activate train_abc

python preprocess.py
