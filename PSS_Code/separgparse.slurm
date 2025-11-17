#!/bin/bash
#SBATCH --cpus-per-task=94
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=windfall
#SBATCH	--mail-type=END
#SBATCH	--mail-user=benyurek@arizona.edu
#SBATCH --array=1-51

module load python/3.9/3.9.10
source /home/u23/benyurek/env39/bin/activate

FileIdNum=$((SLURM_ARRAY_TASK_ID-1))

PathInit="/xdisk/hamden/benyurek/env39"
CreatePath=""${PathInit}"/raw_files/Batch_"${FileIdNum}""
CurrentFile=""${PathInit}"/input/coord_groups/group"${FileIdNum}"_res4.txt"

mkdir $CreatePath 2>/dev/null || (rm -rf $CreatePath && mkdir $CreatePath)

for line in {1..4}
do
    CurrentCoordinates="$( sed "${line}q;d" $CurrentFile )"
    date
    echo "Before Query"
    time srun python3 query.py --pair-list $CurrentCoordinates
    date
    echo "Before Preprocessing"
    time srun python3 preprocessing.py --pair-list $CurrentCoordinates
    date
    echo "Before obj_dec"
    time srun python3 object_detection.py --pair-list $CurrentCoordinates
    date
    
done