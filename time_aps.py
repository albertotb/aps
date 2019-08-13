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
import math

def optimal_number_iters(d_values, a_values, d_true, disc, times=50, n_jobs=-1):

    iter_list = [100, 500, 1000, 2000, 3000, 4000]
    temp_list  = [10, 100, 1000, 2000, 3000, 4000]

    for temp, iters in product(temp_list,iter_list):
        def find_d_opt():
            d_opt = aps_adg_ann(temp, temp, p.d_util, p.a_util, p.prob,
                                N_aps=iters, N_inner=iters, burnin=0.5,
                                prec=disc, mean=True, info=False)
            return d_opt

        optimal_d = Parallel(n_jobs=n_jobs)(
            delayed(find_d_opt)() for j in range(times)
        )

        optimal_d = np.round(optimal_d, int( -np.log10(disc) ) )
        percent = np.mean( np.isclose( np.array(optimal_d), d_true ) )

        # percent = np.mean(
        #    np.isclose(np.array(optimal_d), d_true, rtol=disc)
        #)
        ## Are 90% equal to the truth? Then we converge.
        if percent >= 0.9:
            break

    return iters, temp


if __name__ == '__main__':

    disc_list = np.array([0.1, 0.01])

    results = []
    for disc in disc_list:
        a_values = np.arange(0, 1, disc)
        d_values = np.arange(0, 1, disc)

        d_true = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                          p.prob, mcmc_iters=100000, info=False)

        iters, temp = optimal_number_iters(d_values, a_values, d_true, disc,
                                           n_jobs=6)

        start = default_timer()
        d_opt = aps_adg_ann(temp, temp, p.d_util, p.a_util, p.prob,
                            N_aps=iters, N_inner=iters, burnin=0.5,
                            prec=disc, mean=True, info=False)
        end = default_timer()
        time = end-start

        results.append({'disc': disc, 'time': time,
                        'iters': iters, 'temp': temp,
                        'd_true': d_true, 'd_opt': d_opt})

        print(disc, time, iters, temp, d_true, d_opt)

    df = pd.DataFrame(results)
    df.to_csv('results/times_aps_09_new.csv', index=False)