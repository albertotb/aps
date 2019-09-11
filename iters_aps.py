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

def optimal_number_iters(d_values, a_values, d_true, disc, times=10, n_jobs=-1):

    params = {'J_inner': [10, 40, 100, 1000, 10000],            # 40    1000
              'J':       [10, 40, 1000, 100000, 200000],        # 40  100000
              'N_inner': [10, 50, 100, 1000],                   # 50     100
              'N_aps':   [10, 50, 100, 1000, 2000, 10000]}      # 50    1000

    # the df has to be sorted in the product from less impact to more
    # impact in the complexity of the algorithm
    param_df = (pd.DataFrame(product(*params.values()), columns=params.keys())
                  .sort_values(by=['N_aps','N_inner', 'J', 'J_inner']))

    for _, param in param_df.iterrows():

        def find_d_opt():
            d_opt = aps_adg_ann(p.d_util, p.a_util, p.prob, burnin=0.5,
                                prec=disc, mean=True, info=False, **param)
            return d_opt

        optimal_d = Parallel(n_jobs=n_jobs)(
            delayed(find_d_opt)() for j in range(times)
        )

        optimal_d = np.round(optimal_d, int(-np.log10(disc)))
        percent = np.mean(np.isclose(np.array(optimal_d), d_true, rtol=disc))

        # percent = np.mean(
        #    np.isclose(np.array(optimal_d), d_true, rtol=disc)
        #)
        ## Are 90% equal to the truth? Then we converge.
        if percent >= 0.9:
            break

    return temp_outer, temp_inner, iter_outer, iter_inner


if __name__ == '__main__':

    disc_list = np.array([0.1, 0.01])

    results = []
    for disc in disc_list:
        a_values = np.arange(0, 1, disc)
        d_values = np.arange(0, 1, disc)

        #d_true = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
        #                  p.prob, mcmc_iters=1000000, info=False)

        d_true = 1
        temp_outer, temp_inner, iter_outer, iter_inner = optimal_number_iters(
                d_values, a_values, d_true, disc, n_jobs=10)

        break
        start = default_timer()
        d_opt = aps_adg_ann(temp_outer, temp_inner, p.d_util, p.a_util, p.prob,
                            N_aps=iter_outer, N_inner=iter_inner, burnin=0.5,
                            prec=disc, mean=True, info=False)
        end = default_timer()
        time = end-start

        results.append({'disc': disc, 'time': time, 'iter_outer': iter_outer,
                        'iter_inner': iter_inner, 'temp_outer': temp_outer,
                        'temp_inner': temp_inner, 'd_true': d_true, 
                        'd_opt': d_opt})

        print(disc, time, iter_outer, iter_inner, temp_outer, temp_inner,
              d_true, d_opt)

    df = pd.DataFrame(results)
    df.to_csv('results/iters_aps.csv', index=False)
