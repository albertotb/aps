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

    parser.add_argument('--aps',
                type=int,
                dest='aps',
                help='Number of outer APS iterations',
                default=10000)

    parser.add_argument('--aps_inner',
                type=int,
                dest='aps_inner',
                help='Number of inner APS iterations',
                default=1000)

    parser.add_argument('--burnin',
                type=float,
                dest='burnin',
                help='Percentage of iterations to discard',
                default=0.75)

    args = parser.parse_args()

    p = import_module(args.module)

    np.random.seed(1234)

    print('APS ADG')
    with timer():
        d_opt, a_opt = aps_atk_def(p.d_values, p.a_values, p.d_util, p.a_util,
                                   p.prob, N_aps=args.aps, burnin=args.burnin,
                                   N_inner=args.aps_inner)
    print(d_opt)
    print(a_opt)
