# Augmented Probability Simulation

Code to run the simulations in the paper "*Augmented Probability Simulation Methods for Non-cooperative Games*".

The folder structure is the following:

  * `notebook/`, Jupyter notebooks with analysis
  * `results/`, output files
  * `aps.py`, implementation of the APS algorithm
  * `mcmc.py`, implementation of the MCMC algorithm
  * `prob1.py`, information specific to example 1
  * `prob2.py`, information specific to example 2
  * `run_exp.py`, interface to run the experiments
  * `run_exp.sh`, wrapper to run the experiments in a SGE cluster
  * `run_all.sh`, helper script to run all experiments in a SGE cluster
  
 Run the experiments in `lovelace` using MCMC: 
 
     ./run_all.sh mcmc
     
 Run the experiments in `lovelace` using APS:
 
     ./run_all.sh aps
     
 Run all experiments:
 
     ./run_all.sh
