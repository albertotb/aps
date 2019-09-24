# Augmented Probability Simulation

Code to run the simulations in the paper "*Augmented Probability Simulation Methods for Non-cooperative Games*".

The folder structure is the following:

  * `notebook/`, Jupyter notebooks with analysis
  * `results/`, output files
  * `data/`, problem data files
  * `img/`, figures
  * `aps.py`, implementation of the APS algorithm
  * `aps_annealing.py`, implementation of the APS annealing strategy
  * `mcmc.py`, implementation of the MCMC algorithm
  * `opt_iters.py`, script to compute optimal number of iterations
  * `plot.R`, script to generate all the plots
  * `sensitivity_analysis.py`, script to perform sensitivity analysis of problem 1
  * `true_sol.py`, script to compute the true solution of a problem
  * `run_exp.py`, interface to run the experiments
  * `run_sge.sh`, wrapper to run the experiments in a SGE cluster
  * `run_all.sh`, helper script to run all experiments
  
 Run the experiments in `lovelace` using MCMC: 
 
     ./run_all.sh mcmc
     
 Run the experiments in `lovelace` using APS:
 
     ./run_all.sh aps
     
 Run all experiments:
 
     ./run_all.sh
