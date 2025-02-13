#!/bin/bash

# SLURM parameters
#SBATCH -n 1
#SBATCH -t 5
#SBATCH --mem-per-cpu 3800
#SBATCH --gpus-per-task=1

# Special parameters. DO NOT CHANGE THESE!
#SBATCH -A kurs00082
#SBATCH -p kurs00082
#SBATCH --reservation=kurs00082

# Redirect stdout and stderr
#SBATCH -o ex1.out
#SBATCH -e ex1.err

module purge
module load cuda/12.5 gcc/13.1.0
mkdir -p out
srun build/pmpp_executable
