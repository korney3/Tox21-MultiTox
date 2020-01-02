#!/bin/bash -l
module purge
module load python/pytorch-1.3.0
module rm python/python-3.6.8
module load python/python-3.7.1
module load gpu/cuda-10.0
module list

python Neural_Net_sigma_train_optimization.py -n 1 > logs.txt
#python python_test.py > logs.txt
#nvidia-smi
#python "print('hello world')"
#./slurm_test.sh
#python ./python_test.py
#python3 ./python_test.py
