#!/bin/bash -l 

#SBATCH --mem=60000

module load python/python-3.7.1
module load gpu/cuda-10.0
module load python/pytorch-1.3.0
module rm python/python-3.6.8
module list

python Neural_Net_classification_and_regression.py -m $m -n $n -b $b -t $t -s $s -c $c -l $l -v 70

