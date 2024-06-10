#!/bin/bash
#SBATCH --job-name=memory3mil         # Job name
#SBATCH --mail-type=ALL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=email@du.edu     # Where to send mail
#SBATCH --ntasks=1                   # a single CPU
#SBATCH --mem=10gb                    # Job memory request
#SBATCH --time=02:05:00              # Time limit hrs:min:sec
#SBATCH --output=job_%j.log    # Standard output and error log
#SBATCH --exclude=node001,node002,node012,node013
 
# Load modules below
module load compilers/anaconda-2021.11
 
# Execute commands for application below
python Metrogeneral.py
