#!bin/bash

#PBS -N 3c66b_11_mc_restricted
#PBS -q sbs0016
#PBS -l nodes=1:ppn=2,pvmem=12gb,walltime=168:00:00
#PBS -m abe
#PBS -M bdc0001@mix.wvu.edu
#PBS -e /scratch/bdc0001/jobs_out/3c66b_11_mc_restricted.err
#PBS -o /scratch/bdc0001/jobs_out/3c66b_11_mc_restricted.out

FORB=-6.713761323432285
source /users/bdc0001/miniconda3/etc/profile.d/conda.sh
conda activate enterprise_pls_work
cd /users/bdc0001/enterprise_extensions/enterprise_extensions/deterministic/eccentric_search/ecc_search_code/
python ecc_search_fixed_forb_trunc2.py --gwphi 0.62478505352 \
 --gwtheta 0.75035284894 \
 --gwdist 7.929418925714293 \
 --inc 1.5707963267948966 \
 -f ${FORB} \
 --datadir /scratch/bdc0001/partim_new/ \
 --noisedir /scratch/bdc0001/noisefiles_new/ \
 --crn_type fixed \
 --psrs \
 --outdir /scratch/bdc0001/ecc_search_data/detection_runs/targeted/3c66b/truncated/no_1713/fixed_crn/mc_restricted/11y/run1/ \
 --pkl 11y_partim_new.pkl \
 --rn_pkl rn_distr.pkl
