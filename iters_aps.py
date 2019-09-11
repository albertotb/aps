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

    temp_inner_list = [100]       # 40    1000
    temp_outer_list = np.arange(1000, 10000, 100)#[10, 40, 1000, 100000, 200000]   # 40  100000
    iter_inner_list = [10, 50, 100]              # 50     100
    iter_outer_list = [300, 400, 500]       # 50    1000

    # the lists have to be sorted in the product from less impact to more
    # impact in the complexity of the algorithm
    prd = product(iter_outer_list, temp_outer_list, iter_inner_list, temp_inner_list)
    for iter_outer, temp_outer, iter_inner, temp_inner in prd:
        def find_d_opt():
            d_opt = aps_adg_ann(temp_outer, temp_inner, p.d_util, p.a_util,
                                p.prob, N_aps=iter_outer, N_inner=iter_inner,
                                burnin=0.05, prec=disc, mean=True, info=False)
            return d_opt

        optimal_d = Parallel(n_jobs=n_jobs)(
            delayed(find_d_opt)() for j in range(times)
        )

        optimal_d = np.round(optimal_d, int(-np.log10(disc)))
        percent = np.mean(np.isclose(np.array(optimal_d), d_true, rtol=disc))

        print("iter_outer: " + str(iter_outer) + "temp_outer: "+ str(temp_outer) + "iter_inner: " + str(iter_inner) + "temp_inner: " + str(temp_inner))
        print("percent:", percent)
        # percent = np.mean(
        #    np.isclose(np.array(optimal_d), d_true, rtol=disc)
        #)
        ## Are 90% equal to the truth? Then we converge.
        if percent >= 0.9:
            break

    return temp_outer, temp_inner, iter_outer, iter_inner


if __name__ == '__main__':

    disc_list = np.array([0.01])

    results = []
    for disc in disc_list:
        a_values = np.arange(0, 1, disc)
        d_values = np.arange(0, 1, disc)

        #d_true = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
        #                  p.prob, mcmc_iters=1000000, info=False)
        d_true = 0.46
        temp_outer, temp_inner, iter_outer, iter_inner = optimal_number_iters(
                d_values, a_values, d_true, disc, n_jobs=10)

        start = default_timer()
        d_opt = aps_adg_ann(temp_outer, temp_inner, p.d_util, p.a_util, p.prob,
                            N_aps=iter_outer, N_inner=iter_inner, burnin=0.2,
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
    df.to_csv('results/iters_aps_roi_2.csv', index=False)
