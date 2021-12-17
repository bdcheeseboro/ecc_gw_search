#!/bin/bash

OUTDIR='/home/bcheeseboro/nanograv_proj/enterprise_proj/ecc_signal_create/ecc_sim_data/fixed_coords/correct_dist/efac_added/logmc_9.5/source11/'
PARFILES=(${OUTDIR}'*.par')
SUB='DMX_'

echo ${PARFILES}
for pfile in ${PARFILES[@]} ; do
	this_file=$($pfile)
	echo ${this_file}
	#if [[ "$STR" =~ .*"$SUB".* ]]; then
  #echo "It's there."
#fi
done
