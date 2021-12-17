#!bin/bash

#PBS -N 3c66b_12_inj_dist_further2
#PBS -q sbs0016
#PBS -l nodes=1:ppn=4,walltime=168:00:00
#PBS -m abe
#PBS -M bdc0001@mix.wvu.edu
#PBS -e /scratch/bdc0001/jobs_out/3c66b_12_inj_dist_further2.err
#PBS -o /scratch/bdc0001/jobs_out/3c66b_12_inj_dist_further2.out

FORB=-6.713761323432285
source /users/bdc0001/miniconda3/etc/profile.d/conda.sh
conda activate enterprise_pls_work
cd /users/bdc0001/enterprise_extensions/enterprise_extensions/deterministic/eccentric_search/ecc_search_code/
python ecc_search_ideal_fixed_forb.py --gwphi 0.62478505352  \
 --gwtheta 0.75035284894 \
 --gwdist 7.929418925714293  \
 --inc 1.0471975512 \
 -f ${FORB} \
 --datadir /scratch/bdc0001/ecc_signal_create/ecc_sim_data/12p5_simulated/logmc_8.5/source3/ \
 --noisedir /scratch/bdc0001/ \
 --outdir /scratch/bdc0001/ecc_search_data/injection_runs/12p5y/logmc_8.5/source3/run1/ \
 --pkl channelized_12yr_v3_partim_py3.pkl \
 --rn_pkl 12yr_emp_dist_RNonly_py3.pkl
