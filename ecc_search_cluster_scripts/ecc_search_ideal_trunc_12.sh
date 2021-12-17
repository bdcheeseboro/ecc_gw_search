#!bin/bash

#PBS -N 3c66b_12_inj_redo_3
#PBS -q sbs0016
#PBS -l nodes=1:ppn=4,walltime=168:00:00
#PBS -m abe
#PBS -M bdc0001@mix.wvu.edu
#PBS -e /scratch/bdc0001/jobs_out/3c66b_12_inj_redo_3.err
#PBS -o /scratch/bdc0001/jobs_out/3c66b_12_inj_redo_3.out

FORB=-8.5
source /users/bdc0001/miniconda3/etc/profile.d/conda.sh
conda activate enterprise_pls_work
cd /users/bdc0001/enterprise_extensions/enterprise_extensions/deterministic/eccentric_search/ecc_search_code/
python ecc_search_ideal_fixed_forb.py --gwphi 5.01  \
 --gwtheta 1.91 \
 --gwdist 7.5  \
 --inc 1.0471975512 \
 -f ${FORB} \
 --datadir /scratch/bdc0001/ecc_signal_create/ecc_sim_data/12p5_simulated/logmc_9.5/source2/ \
 --noisedir /scratch/bdc0001/ \
 --outdir /scratch/bdc0001/ecc_search_data/injection_runs/mc_e0_restricted/12p5y/logmc_9.5/source2/run1/ \
 --pkl ideal_pulsars_ecc_search.pkl \
 --rn_pkl 12yr_emp_dist_RNonly_py3.pkl
