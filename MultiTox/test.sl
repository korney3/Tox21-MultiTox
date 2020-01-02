#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=example


# run for five minutes
#              d-hh:mm:ss
#SBATCH --time=0-00:05:00

# 500MB memory per core
# this is a hard limit

# you may not place bash commands before the last SBATCH directive
module purge
module load python/pytorch-1.3.0
module rm python/python-3.6.8
module load python/python-3.7.1
module load gpu/cuda-10.0
module list

# define and create a unique scratch directory
echo "now processing task id:: "
python python_test.py > logs.txt
nvidia-smi
# after the job is done we copy our output back to $SLURM_SUBMIT_DIR


# we step out of the scratch directory and remove it
# happy end
exit 0
