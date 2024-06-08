#!/bin/bash
#SBATCH --job-name=mc100
#SBATCH --output=out.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 8-0:00 
#SBATCH --array=0-25





#    #####SBATCH --exclude=node001,node002,node012,node013
# Load modules below
#module load compilers/anaconda-3.8-2020.11 
module load compilers/anaconda-2021.11
#module load gnu-parallel/2021.07.22
# Execute commands for application below
#ideal 128 tasks 4 nodes
​
python infer_fast.py $1 $SLURM_ARRAY_TASK_ID
​
​
