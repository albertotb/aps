#!/usr/bin/env python
import numpy as np
import pandas as pd
import pickle
import prob_new as p
from aps_annealing import *
from mcmc import *
from joblib import Parallel, delayed
import multiprocessing
import sys
from timeit import default_timer
from itertools import product

def optimal_number_iters(d_values, a_values, d_true, times=50, n_jobs=-1):

    iters_list = np.arange(1000, 10000000, 1000)
    inner_iters_list = np.arange(100, 100000, 500)
    #
    for inner, iters in product(inner_iters_list, iters_list):

        def find_d_opt_MC(j):
            d_opt = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob, 
                             p.prob, mcmc_iters=iters, inner_mcmc_iters=inner, info=False)
            return d_opt

        optimal_d = Parallel(n_jobs=n_jobs)(
            delayed(find_d_opt_MC)(j) for j in range(times)
        )

        percent = np.mean( np.isclose( np.array(optimal_d), d_true ) )
        ## Are 90% equal to the truth? Then we converge.
        if percent >= 0.9:
            break

    return iters, inner

if __name__ == '__main__':

    disc_list = np.array([0.1, 0.01])

    results = []
    for disc in disc_list:
        a_values = np.arange(0, 1, disc)
        d_values = np.arange(0, 1, disc)

        d_true = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                          p.prob, mcmc_iters=100000, info=False)

        iters, inner = optimal_number_iters(d_values, a_values, d_true, n_jobs=8)

        start = default_timer()
        d_opt = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                         p.prob, mcmc_iters=iters, mcmc_iters_inner = inner, info=False)
        end = default_timer()
        time = end-start

        results.append({'disc': disc, 'time': time, 'iters': iters, 'inner': inner,
                        'd_true': d_true, 'd_opt': d_opt})

        print(disc, time, iters, inner, d_true, d_opt)

    df = pd.DataFrame(results)
    df.to_csv('results/times_mc_0.9_new.csv', index=False)