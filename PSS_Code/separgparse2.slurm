#!/bin/bash
#SBATCH --cpus-per-task=94
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --partition=windfall
#SBATCH	--mail-type=END
#SBATCH	--mail-user=benyurek@arizona.edu
#SBATCH --array=1-4

module load julia
source /home/u23/benyurek/env39/bin/activate
export JULIA_NUM_THREADS=10

group_num=$1
CurrentFile="/xdisk/hamden/benyurek/env39/input/coord_groups/group${group_num}_res4.txt"

echo "$CurrentFile"

date
srun julia pointsource_infill.jl $SLURM_ARRAY_TASK_ID $CurrentFile
date
