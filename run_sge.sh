#!/bin/bash
#$ -cwd
#$ -j y
#$ -q all.q
#$ -pe omp 16
python $@
