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
from datetime import datetime
import math

BURNIN = 0.2
TIMES = 10
N_JOBS = 10
ITERS_TRUE_SOL = 1000000

def optimal_number_iters(d_values, a_values, d_true, disc, times=10, n_jobs=1):

    params = {'J_inner': [10, 50, 100, 1000, 2000],      # 40    1000
              'J':       np.arange(1000, 11000, 1000),   # 40  100000
              'N_inner': [10, 50, 100, 1000, 2000],      # 50     100
              'N_aps':   [10, 50, 100, 1000, 2000]}      # 50    1000

    # the df has to be sorted in the product from less impact to more
    # impact in the complexity of the algorithm
    param_df = (pd.DataFrame(product(*params.values()), columns=params.keys())
                  .sort_values(by=['N_aps', 'N_inner', 'J', 'J_inner']))

    for _, param in param_df.iterrows():

        def find_d_opt():
            d_opt = aps_adg_ann(p.d_util, p.a_util, p.prob, burnin=BURNIN,
                                prec=disc, mean=True, info=False, **param)
            return d_opt

        optimal_d = Parallel(n_jobs=n_jobs)(
            delayed(find_d_opt)() for j in range(times)
        )

        optimal_d = np.round(optimal_d, int(-np.log10(disc)))
        percent = np.mean(np.isclose(np.array(optimal_d), d_true, rtol=disc))

        ## Are 90% equal to the truth? Then we converge.
        if percent >= 0.9:
            break

    return param


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('usage: {} PREC...'.format(sys.argv[0]))
        sys.exit(1)

    ts = datetime.now().timestamp()
    disc_list = list(map(float, sys.argv[1:]))

    results = []
    for disc in disc_list:
        a_values = np.arange(0, 1, disc)
        d_values = np.arange(0, 1, disc)

        d_true = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                          p.prob, mcmc_iters=ITERS_TRUE_SOL, info=False)

        params = optimal_number_iters(d_values, a_values, d_true, disc,
                                      times=TIMES, n_jobs=N_JOBS)

        start = default_timer()
        d_opt = aps_adg_ann(p.d_util, p.a_util, p.prob, burnin=BURNIN,
                            prec=disc, mean=True, info=False, **params)
        end = default_timer()
        time = end-start

        results.append({'timestamp': ts, 'disc': disc, 'd_true': d_true,
                        'd_opt': d_opt, 'time': time, 'burnin': BURNIN,
                        'times': TIMES, 'iters_true_sol': ITERS_TRUE_SOL,
                        **params})

    df = pd.DataFrame(results).set_index(['timestamp', 'disc'])
    df.to_csv('results/iters_aps.csv', mode='a')
