#!/bin/bash
#SBATCH --cpus-per-task=94
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --partition=windfall
#SBATCH	--mail-type=END
#SBATCH	--mail-user=benyurek@arizona.edu
#SBATCH --array=1-4

module load python/3.9
source /home/u23/benyurek/env39/bin/activate

group_num=$1
CurrentFile="/xdisk/hamden/benyurek/env39/input/coord_groups/group${group_num}_res4.txt"
CurrentCoordinates="$( sed "${SLURM_ARRAY_TASK_ID}q;d" $CurrentFile )"

echo "$CurrentFile"

date
time srun python3 montage.py --pair-list $CurrentCoordinates
date
