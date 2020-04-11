#!/bin/bash -l

#SBATCH --mem=20000


module load python/python-3.7.1
module load gpu/cuda-10.0
module load python/pytorch-1.3.0
module rm python/python-3.6.8
module list

python Input_backprop.py -n 23
