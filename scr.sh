#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Partition:
#SBATCH --partition=debug
#
# Number of MPI tasks needed for use case (example):
#SBATCH --ntasks=2
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
## Command(s) to run (example):
conda activate pydeph
mpirun -np 2 python pydephasing --spin --inhomo --stat input_T2st.yml
sleep 5
