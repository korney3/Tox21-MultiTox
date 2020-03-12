#!/bin/bash
#SBATCH -o mpi_run_database.out
#SBATCH -e mpi_run_database.err
#SBATCH --partition cpu_small
#SBATCH --job-name molecules_sql_calc

#SBATCH -N 1
#SBATCH --ntasks 24

module load python/python-3.7.1
#module load mpi/openmpi-3.1.4
module load mpi/hpcx-v2.5.0
srun -n $SLURM_NTASKS python sql_database_creating_multitox_more_atoms.py -f 'MultiTox0'
