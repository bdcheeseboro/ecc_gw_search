#!/bin/bash

#PBS -N ecc_sim_make_11y
#PBS -q sbs0016
#PBS -l nodes=1:ppn=4
#PBS -m abe
#PBS -M bdc0001@mix.wvu.edu
#PBS -e /scratch/bdc0001/jobs_out/ecc_sim_make_11y.err
#PBS -o /scratch/bdc0001/jobs_out/ecc_sim_make_11y.out

source /users/bdc0001/miniconda3/etc/profile.d/conda.sh
echo "activating conda env"
conda activate enterprise
echo "entering code dir"
cd /users/bdc0001/enterprise_extensions/enterprise_extensions/deterministic/eccentric_search/ecc_search_code/
echo "running py script"
#python hello_world.py
python ecc_res_simulate.py --datadir /scratch/bdc0001/partim_new/ --noisedir /scratch/bdc0001/noisefiles_new/ --outdir /scratch/bdc0001/ecc_signal_create/ecc_sim_data/11_simulated/logmc_9.5/source1/ --psrs True
echo "run completed"