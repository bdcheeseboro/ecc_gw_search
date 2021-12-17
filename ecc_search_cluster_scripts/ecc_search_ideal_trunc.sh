#!bin/bash

#PBS -N 3c66b_11_inj
#PBS -q sbs0016
#PBS -l nodes=1:ppn=2,pvmem=12gb,walltime=168:00:00
#PBS -m abe
#PBS -M bdc0001@mix.wvu.edu
#PBS -e /scratch/bdc0001/jobs_out/3c66b_11_inj.err
#PBS -o /scratch/bdc0001/jobs_out/3c66b_11_inj.out

FORB=-8.5
source /users/bdc0001/miniconda3/etc/profile.d/conda.sh
conda activate enterprise
cd /users/bdc0001/enterprise_extensions/enterprise_extensions/deterministic/eccentric_search/ecc_search_code/
python ecc_search_ideal_fixed_forb.py --gwphi 5.01 \
 --gwtheta 1.91 \
 --gwdist 7.5 \
 --inc 1.0471975512 \
 -f ${FORB} \
 --datadir /scratch/bdc0001/ecc_signal_create/ecc_sim_data/11_simulated/logmc_9.5/source1/ \
 --noisedir /scratch/bdc0001/noisefiles_new/ \
 --outdir /scratch/bdc0001/ecc_search_data/injection_runs/11y/logmc_9.5/source1/run1/ \
 --pkl 11y_partim_new.pkl \
 --rn_pkl rn_distr.pkl \
 --psrs_11
