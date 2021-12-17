#!/bin/bash

#PBS -N ecc_sim_make_logmc_8.5
#PBS -q sbs0016
#PBS -l nodes=1:ppn=4
#PBS -m abe
#PBS -M bdc0001@mix.wvu.edu
#PBS -e /scratch/bdc0001/jobs_out/ecc_source3.err
#PBS -o /scratch/bdc0001/jobs_out/ecc_source3.out

source /users/bdc0001/miniconda3/etc/profile.d/conda.sh
echo "activating conda env"
conda activate enterprise_pls_work
echo "entering code dir"
cd /users/bdc0001/enterprise_extensions/enterprise_extensions/deterministic/eccentric_search/ecc_search_code/
echo "running py script"
#python hello_world.py
python ecc_res_simulate.py \
 --datadir /scratch/bdc0001/NANOGrav_12yv4/narrowband/ \
 --noisedir /scratch/bdc0001/ \
 --outdir /scratch/bdc0001/ecc_signal_create/ecc_sim_data/12p5_simulated/logmc_8.5/source3/
echo "run completed"