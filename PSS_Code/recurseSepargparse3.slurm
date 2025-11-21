#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --partition=windfall
#SBATCH	--mail-type=END
#SBATCH	--mail-user= !!!!!


echo "Executing separgparse 3 with the iteration argument : "
echo $1

arg_int=$1
incremented_arg=$((arg_int + 1))

if [ "$arg_int" -le 767 ]; then
    sbatch newsepargparse3.slurm $arg_int
    sbatch recurseSepargparse3.slurm $incremented_arg
else
    echo "limit reached...no more jobs to call"
fi
