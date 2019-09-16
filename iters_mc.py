#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import pickle
import prob_new as p
from mcmc import *
from joblib import Parallel, delayed
import multiprocessing
import sys
from timeit import default_timer
from itertools import product
from datetime import datetime

PER_TIMES = 0.9
TIMES = 50
N_JOBS = 10
ITERS_TRUE_SOL = 1000000

def optimal_number_iters(d_values, a_values, d_true, times=10, n_jobs=1):

    iters_list = np.arange(1000, 10000000, 1000)
    inner_iters_list = np.arange(100, 100000, 500)

    for inner, iters in product(inner_iters_list, iters_list):

        def find_d_opt(j):
            d_opt = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                             p.prob, iters=iters, inner_iters=inner, info=False)
            return d_opt

        optimal_d = Parallel(n_jobs=n_jobs)(
            delayed(find_d_opt)(j) for j in range(times)
        )

        percent = np.mean( np.isclose( np.array(optimal_d), d_true ) )
        ## Are 90% equal to the truth? Then we converge.
        if percent >= PER_TIMES:
            break

    return iters, inner


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('usage: {} PREC..'.format(sys.argv[0]))
        sys.exit(1)

    fout = 'results/iters_mc.csv'
    ts = datetime.now().timestamp()
    disc_list = list(map(float, sys.argv[1:]))

    results = []
    for disc in disc_list:
        a_values = np.arange(0, 1, disc)
        d_values = np.arange(0, 1, disc)

        d_true = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                          p.prob, iters=ITERS_TRUE_SOL, info=False)

        iters, inner = optimal_number_iters(d_values, a_values, d_true,
                                            n_jobs=N_JOBS, times=TIMES)

        start = default_timer()
        d_opt = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                         p.prob, iters=iters, inner_iters=inner, info=False)
        end = default_timer()
        time = end-start

        results.append({'timestamp': ts, 'disc': disc, 'time': time,
                        'd_true': d_true, 'd_opt': d_opt, 'times': TIMES,
                        'per_times': PER_TIMES,
                        'iters_true_sol': ITERS_TRUE_SOL,
                        'iters': iters, 'inner': inner})

    header = not (os.path.exists(fout) and os.path.getsize(fout) > 0)
    df = pd.DataFrame(results).set_index(['timestamp', 'disc'])
    df.to_csv(fout, mode='a', header=header)
