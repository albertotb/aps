#!/bin/bash


d=./results/$(date +%Y%m%d_%H%M)

mkdir -p $d

qsub -o $d/prob1_mcmc_adg.out -j y ./run_exp.sh -p prob1 -a mcmc -s adg -o $d --mcmc 100000
qsub -o $d/prob1_mcmc_ara.out -j y ./run_exp.sh -p prob1 -a mcmc -s ara -o $d --mcmc 100000 --ara 1000
qsub -o $d/prob1_aps_adg.out  -j y ./run_exp.sh -p prob1 -a aps  -s adg -o $d --aps  100000 --aps_inner 15000
qsub -o $d/prob1_aps_ara.out  -j y ./run_exp.sh -p prob1 -a aps  -s ara -o $d --aps  100000 --aps_inner 15000 --ara 1000

qsub -o $d/prob2_mcmc_adg.out -j y ./run_exp.sh -p prob2 -a mcmc -s adg -o $d --mcmc 100000
qsub -o $d/prob2_mcmc_ara.out -j y ./run_exp.sh -p prob2 -a mcmc -s ara -o $d --mcmc 100000 --ara 10000
#qsub -o $d/prob2_aps_adg.out -j y ./run_exp.sh -p prob2 -a aps  -s adg -o $d --aps  100000 --aps_inner 10000
#qsub -o $d/prob2_aps_ara.out -j y ./run_exp.sh -p prob2 -a aps  -s adg -o $d --aps  100000 --aps_inner 10000 --ara 10000
