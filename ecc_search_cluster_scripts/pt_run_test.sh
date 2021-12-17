#!bin/bash

#PBS -N pt_run_ecc3
#PBS -q sbs0016
#PBS -l nodes=1:ppn=20
#PBS -m abe
#PBS -M bdc0001@mix.wvu.edu

source /users/bdc0001/miniconda3/etc/profile.d/conda.sh
conda activate enterprise
cd /users/bdc0001/eccentric_search/ecc_search_code/
mpirun -np 20 python ecc_search_ideal.py
