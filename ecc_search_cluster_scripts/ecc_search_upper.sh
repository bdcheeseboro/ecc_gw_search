#!bin/bash

#PBS -N 3c66b_11_up_fixed_9
#PBS -q sbs0016
#PBS -l nodes=1:ppn=2,pvmem=12gb,walltime=168:00:00
#PBS -m abe
#PBS -M bdc0001@mix.wvu.edu
#PBS -e /scratch/bdc0001/jobs_out/3c66b_11_up_fixed.err
#PBS -o /scratch/bdc0001/jobs_out/3c66b_11_up_fixed.out

FORB=-6.713761323432285
source /users/bdc0001/miniconda3/etc/profile.d/conda.sh
conda activate enterprise
cd /users/bdc0001/enterprise_extensions/enterprise_extensions/deterministic/eccentric_search/ecc_search_code/
python ecc_search_upper.py --gwphi 0.62478505352 \
 --gwtheta 0.75035284894 \
 --gwdist 7.929418925714293 \
 --inc 1.5707963267948966 \
 -f ${FORB} \
 --datadir /scratch/bdc0001/partim_new/ \
 --noisedir /scratch/bdc0001/noisefiles_new/ \
 --crn_type fixed \
 --outdir /scratch/bdc0001/ecc_search_data/upper_limit_runs/targeted/3c66b/fixed_crn/11y/run9/ \
## --bayesephem \
 --pkl 11y_partim_new.pkl \
 --rn_pkl rn_distr.pkl
