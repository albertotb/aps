#!/usr/bin/env python
import os
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

BURNIN = 0.05
PER_TIMES = 0.9
TIMES = 50
N_JOBS = 16
ITERS_TRUE_SOL = 1000000

def optimal_number_iters(alg, d_values, a_values, d_true, disc, times=10,
                         n_jobs=1):

    # the df has to be sorted in the product from less impact to more
    # impact in the complexity of the algorithm
    if alg == 'mcmc':
        params = {'iters': np.arange(1000, 51000, 1000),
                  'inner_iters': np.arange(100, 11000, 100)}

        param_df = (pd.DataFrame(product(*params.values()), columns=params.keys())
                      .sort_values(by=['iters', 'inner_iters']))

    elif alg == 'aps':
        params = {'J_inner': [10, 50, 100],
                  'J':       np.arange(10, 110, 10) * int(1/disc),
                  'N_inner': [10, 50, 100],
                  'N_aps':   np.arange(1, 11, 1) * int(1/disc)}

        param_df = (pd.DataFrame(product(*params.values()), columns=params.keys())
                      .sort_values(by=['N_aps', 'J', 'N_inner', 'J_inner']))

    else:
        raise ValueError


    for _, param in param_df.iterrows():

        def find_d_opt():
            if alg == 'mcmc':
                d_opt = mcmc_adg(d_values, a_values, p.d_util, p.a_util,
                                 p.prob, p.prob, info=False, **param)

            elif alg == 'aps':
                d_opt = aps_adg_ann(p.d_util, p.a_util, p.prob, burnin=BURNIN,
                                    prec=disc, mean=True, info=False, **param)

            else:
                raise ValueError

            return d_opt

        optimal_d = Parallel(n_jobs=n_jobs)(
            delayed(find_d_opt)() for j in range(times)
        )

        if alg == 'aps':
            optimal_d = np.round(optimal_d, int(-np.log10(disc)))

        percent = np.mean(np.isclose(np.array(optimal_d), d_true, rtol=disc))

        ## Are 90% equal to the truth? Then we converge.
        if percent >= PER_TIMES:
            break

    return param


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('usage: {} ALG PRECISION..'.format(sys.argv[0]))
        sys.exit(1)

    fout = 'results/opt_iters.csv'
    ts = datetime.now().timestamp()
    alg = sys.argv[1]
    disc_list = list(map(float, sys.argv[2:]))

    results = []
    for disc in disc_list:
        a_values = np.arange(0, 1, disc)
        d_values = np.arange(0, 1, disc)

        start = default_timer()

        d_true = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                          p.prob, iters=ITERS_TRUE_SOL, info=False)

        params = optimal_number_iters(alg, d_values, a_values, d_true, disc,
                                      times=TIMES, n_jobs=N_JOBS)

        end = default_timer()
        time = end-start

        results.append({'timestamp': ts, 'alg': alg, 'disc': disc,
                        'time': time, 'd_true': d_true, 'burnin': BURNIN,
                        'times': TIMES, 'per_times': PER_TIMES,
                        'iters_true_sol': ITERS_TRUE_SOL, **params})

    header = not (os.path.exists(fout) and os.path.getsize(fout) > 0)
    df = pd.DataFrame(results).set_index(['timestamp', 'disc'])
    df.to_csv(fout, mode='a', header=header)
