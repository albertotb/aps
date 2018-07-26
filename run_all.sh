#!/bin/bash


d=$(date +%Y%m%d_%H%M)

mkdir -p $d

#qsub -o $d/prob1_mcmc_adg.out -e $d/prob1_mcmc_adg.err ./run_exp.sh -p prob1 -a mcmc -s atk_def -o $d --mcmc 100000
#qsub -o $d/prob1_mcmc_ara.out -e $d/prob1_mcmc_ara.err ./run_exp.sh -p prob1 -a mcmc -s ara     -o $d --mcmc 100000 --ara 1000
#qsub -o $d/prob1_aps_adg.out  -e $d/prob1_aps_adg.err  ./run_exp.sh -p prob1 -a aps  -s atk_def -o $d --aps  100000 --aps_inner 15000
#qsub -o $d/prob1_aps_ara.out  -e $d/prob1_aps_ara.err  ./run_exp.sh -p prob1 -a aps  -s ara     -o $d --aps  100000 --aps_inner 15000 --ara 1000

qsub -o prob2_mcmc_adg.out -e prob2_mcmc_adg.err ./run_exp.sh -p prob2 -a mcmc -s atk_def --mcmc 100000
#qsub -o prob2_mcmc_ara.out -e prob2_mcmc_ara.err ./run_exp.sh -p prob2 -a mcmc -s ara     --mcmc 100000 --ara 10000
#qsub -o prob2_aps_adg.out  -e prob2_aps_adg.err  ./run_exp.sh -p prob2 -a aps  -s atk_def --aps  100000 --aps_inner 10000
#qsub -o prob2_aps_ara.out  -e prob2_aps_ara.err  ./run_exp.sh -p prob2 -a aps  -s ara     --aps  100000 --aps_inner 10000 --ara 10000
