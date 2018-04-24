#!/usr/bin/env python
import sys
import argparse
import numpy as np
import pandas as pd
from timeit import default_timer
from importlib import import_module
from contextlib import contextmanager

sys.path.append('.')
from mcmc import mcmc_atk_def, mcmc_ara
from aps import aps_atk_def, aps_ara

@contextmanager
def timer():
    start = default_timer()
    try:
        yield
    finally:
        end = default_timer()
        print("Elapsed time (s): {:.6f}".format(end - start))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p',
                dest='module',
                help='module with problem specific information',
                default='prob1',
                choices=['prob1', 'prob2', 'prob1_v2'])

    parser.add_argument('--mcmc',
                type=int,
                dest='mcmc',
                help='Number of MCMC iterations',
                default=10000)

    parser.add_argument('--ara',
                type=int,
                dest='ara',
                help='Number of ARA iterations',
                default=1000)

    args = parser.parse_args()

    p = import_module(args.module)

    np.random.seed(1234)

    print('MCMC ARA')
    with timer():
        d_opt, p_d = mcmc_ara(p.d_values, p.a_values, p.d_util, p.a_util_f,
                              p.prob, p.a_prob_f, n=args.mcmc, m=args.ara)

    df1 = pd.DataFrame(p_d, index=pd.Index(p.d_values, name='d'),
                            columns=pd.Index(p.a_values, name='a'))
    df1.to_pickle('{}_mcmc_pa.pkl'.format(args.module))

    print(d_opt)
    with pd.option_context('display.max_columns', len(p.a_values)):
        print(df1)
