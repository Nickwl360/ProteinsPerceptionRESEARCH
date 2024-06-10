#!/bin/bash
#SBATCH --job-name=milk                        # Job name
#SBATCH --mail-type=NONE       # mail for (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00              # Time limit hrs:min:sec
#SBATCH --output=out.log          # Standard output and error log


module load cuda11.0/toolkit/11.0.3


module load compilers/anaconda-2021.11 



python3 nubtest.py
