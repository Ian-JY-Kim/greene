#!/bin/bash
#
#SBATCH --job-name=myJobarrayTest
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00
#SBATCH --mem=1GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jk7362@nyu.edu

module purge
module load python/intel/3.8.6

cd /home/jk7362/greene
python slurm_test.py $SLURM_ARRAY_TASK_ID