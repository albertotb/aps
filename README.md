# Augmented Probability Simulation

Code to run the simulations in the paper "*Augmented Probability Simulation Methods for Non-cooperative Games*".

The folder structure is the following:

  * `notebook/`, Jupyter notebooks with analysis
  * `results/`, output files
  * `data/`, problem data files
  * `img/`, figures
  * `gif/`, animated figures
  * `aps.py`, implementation of the APS algorithm
  * `aps_annealing.py`, implementation of the APS annealing strategy
  * `mcmc.py`, implementation of the MCMC algorithm
  * `plots.R`, script to generate all the plots
  * `prob1_sa.py`, script to perform sensitivity analysis of problem 1
  * `prob3_opt_iters.py`, script to compute optimal number of iterations
  * `prob3_true_sol.py`, script to compute the true solution of a problem
  * `prob3_runtime.py`, compute running time
  * `aps_ara.py`, run APS for new prob2 ARA
  * `mc_ara.py `, run MC for new prob2 ARA
  * `run_exp.py`, interface to run the experiments

First example problem
---------------------

* Run APS ADG experiment:

      ./run_exp.py mcmc adg --iters 100000

* Run APS ADG experiment:

      ./run_exp.py aps adg --iters 100000 --inner-iters 10000

* Run MC ARA experiment:

      ./run_exp.py mcmc ara --iters 100000 --ara-iters 5000

* Run APS ARA experiment:

      ./run_exp.py aps ara --iters 100000 --inner-iters 10000 --ara-iters 5000

