import numpy as np
import pandas as pd
import prob_new as p
from aps_annealing import *
from mcmc import *
from joblib import Parallel, delayed
import multiprocessing
import sys
from timeit import default_timer
from importlib import import_module
from contextlib import contextmanager

@contextmanager
def timer():
    start = default_timer()
    try:
        yield
    finally:
        end = default_timer()
        print("Elapsed time (s): {:.6f}".format(end - start))

def optimal_number_iters(discretization):
    a_values = np.arange(0, 1, discretization)
    d_values = np.arange(0, 1, discretization)
    ##
    d_true = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob, p.prob,
             mcmc_iters=100000, info=False)
    ##
    iters = np.arange(1000, 10000000, 1000)
    optimal_d = []
    ##
    mylist = np.arange(50)
    ##
    for i in iters:
        ## Run 50 different processes
        num_cores = multiprocessing.cpu_count()
        ##
        def find_d_opt_MC(j):
            d_opt = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob, p.prob,
                     mcmc_iters=i, info=False)
            return d_opt
        ##
        optimal_d = np.asarray( Parallel(n_jobs=num_cores)(delayed(find_d_opt_MC)(j) for j in mylist) )
        ##
        ## Are all equal to the truth? Then we converge.
        if np.all(optimal_d == d_true):
            n_opt_iters = i
            break
        ##
        return n_opt_iters

if __name__ == '__main__':

    disc = np.array([0.1, 0.01, 0.001, 0.0001])
    times = np.zeros_like(disc)
    for i, dd in enumerate(disc):
        n_opt_iters = optimal_number_iters(dd)
        start = default_timer()
        ##
        d_opt = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob, p.prob,
                mcmc_iters=n_opt_iters, info=False)
        ##
        end = default_timer()
        times[i] = end-start

    results = {'Discretization': disc, 'Times': times}
    fout = 'results/times_MC.pkl'
    with open(fout, "wb") as f:
        pickle.dump(results, f)
