#!/bin/bash

d=./results/$(date +%Y%m%d_%H%M)
mkdir -p $d

if [[ -z "$1" || "$1" == "mcmc" ]]; then
    echo "MCMC"
    #qsub -N "mcmc_adg_1" -o $d/prob1_mcmc_adg.out -j y ./run_exp.sh -p prob1 -a mcmc -s adg -o $d --mcmc 100000
    #qsub -N "mcmc_ara_1" -o $d/prob1_mcmc_ara.out -j y ./run_exp.sh -p prob1 -a mcmc -s ara -o $d --mcmc 100000 --ara 10000
    #qsub -N "mcmc_adg_2" -o $d/prob2_mcmc_adg.out -j y ./run_exp.sh -p prob2 -a mcmc -s adg -o $d --mcmc 100000
    qsub -N "mcmc_adg_3" -o $d/prob3_mcmc_adg.out -j y ./run_exp.sh -p prob3 -a mcmc -s adg -o $d --mcmc 100000
    #qsub -N "mcmc_ara_2" -o $d/prob2_mcmc_ara.out -j y -l h_vmem=1G ~/miniconda3/bin/mprof run ./run_exp.sh -p prob2 -a mcmc -s ara -o $d --mcmc 100000 --ara 1000
fi

if [[ -z "$1" || "$1" == "aps" ]]; then
    echo "APS"
    #qsub -N "aps_adg_1" -o $d/prob1_aps_adg.out -j y ./run_exp.sh -p prob1 -a aps  -s adg -o $d --aps  100000 --aps_inner 15000
    #qsub -N "aps_ara_1" -o $d/prob1_aps_ara.out -j y ./run_exp.sh -p prob1 -a aps  -s ara -o $d --aps  100000 --aps_inner 15000 --ara 1000
    #qsub -N "aps_adg_2" -o $d/prob2_aps_adg.out -j y ./run_exp.sh -p prob2 -a aps  -s adg -o $d --aps  100000 --aps_inner 10000
    #qsub -N "aps_ara_2" -o $d/prob2_aps_ara.out -j y ./run_exp.sh -p prob2 -a aps  -s adg -o $d --aps  100000 --aps_inner 10000 --ara 10000
fi
