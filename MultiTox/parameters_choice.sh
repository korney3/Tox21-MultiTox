#!/bin/bash

exp=7

epochs=10000

patience=15

transformations="g w"

learning_rates="0.001 0.0001 0.00001 0.000001"

batch_sizes="10 12 13 15 16"

for tr in g w
do
	for lr in 0.001 0.0001 0.00001 0.000001
	do
		for bs in 10 12 13 15 16
		do
			/home/alisa/anaconda3/bin/python Neural_Net_sigma_train_optimization.py -e $epochs -t $tr -n $exp -b $bs -r $lr -p $patience
			exp=$(($exp+1))
		done
	done
done
