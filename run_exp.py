#!/usr/bin/env python
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from timeit import default_timer
from importlib import import_module
from contextlib import contextmanager

sys.path.append('.')
from mcmc import mcmc_adg, mcmc_ara
from aps import aps_adg, aps_ara
from aps_annealing import aps_adg_ann

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
                choices=['prob1', 'prob2', 'prob3'])

    parser.add_argument('-a',
                dest='alg',
                help='algorithm',
                default='mcmc',
                choices=['mcmc', 'aps', 'aps_annealing'])

    parser.add_argument('-s',
                dest='set',
                help='setting',
                default='adg',
                choices=['adg', 'ara'])

    parser.add_argument('-o',
                dest='out',
                help='output dir',
                default='./results')

    parser.add_argument('--njobs',
                type=int,
                dest='njobs',
                help='Number of jobs',
                default=1)

    parser.add_argument('--mcmc',
                type=int,
                dest='mcmc',
                help='Number of MCMC iterations',
                default=100)

    parser.add_argument('--ara',
                type=int,
                dest='ara',
                help='Number of ARA iterations',
                default=10)

    parser.add_argument('--prob',
                dest='prob',
                help='Probability density p(a | d)',
                default=None)

    parser.add_argument('--aps',
                type=int,
                dest='aps',
                help='Number of outer APS iterations',
                default=100)

    parser.add_argument('--aps_inner',
                type=int,
                dest='aps_inner',
                help='Number of inner APS iterations',
                default=10)

    parser.add_argument('--burnin',
                type=float,
                dest='burnin',
                help='Percentage of iterations to discard',
                default=0.75)

    parser.add_argument('--aps_temp',
                type=int,
                dest='aps_temp',
                help='Temperature for outer and inner annealing aps',
                default=1000)

    parser.add_argument('--mean',
                type=bool,
                dest='aps_mean',
                help='Boolean indicating wether to use the mean or not',
                default=True)

    args = parser.parse_args()

    p = import_module(f'data.{args.module}')

    #np.random.seed(1234)

    d_idx = pd.Index(p.d_values, name='d')
    a_idx = pd.Index(p.a_values, name='a')

    #---------------------------------------------------------------------------
    # Attacker-defender game
    #---------------------------------------------------------------------------
    if args.set == 'adg':
        print('=' * 30)
        print('Game theory')
        print('=' * 30)

        if args.alg == 'mcmc':
            print('MCMC')
            print('-' * 80)
            print('Iters: {}'.format(args.mcmc))
            with timer():
                d_opt, a_opt, psi_d, psi_a, t = mcmc_adg(p.d_values, p.a_values,
                                                         p.d_util, p.a_util,
                                                         p.prob, p.prob,
                                                         mcmc_iters=args.mcmc)
                # print('Elapsed time per attack: ', t)

            psi_d = pd.Series(psi_d, index=d_idx)
            psi_a = pd.DataFrame(psi_a, index=d_idx, columns=a_idx)

        elif args.alg == 'aps':
            print('APS')
            print('-' * 80)
            print('Outer iters: {}'.format(args.aps))
            print('Inner iters: {}'.format(args.aps_inner))
            print('Burnin: {}'.format(args.burnin))
            with timer():
                d_opt, a_opt, psi_d, psi_a = aps_adg(p.d_values, p.a_values,
                                                     p.d_util, p.a_util,
                                                     p.prob, N_aps=args.aps,
                                                     burnin=args.burnin,
                                                     N_inner=args.aps_inner)

            psi_d = pd.Series(psi_d)
            psi_a = pd.Series(psi_a, index=d_idx)

        elif args.alg == 'aps_annealing':
            print('APS ANNEALING')
            print('-' * 80)
            print('Outer iters: {}'.format(args.aps))
            print('Inner iters: {}'.format(args.aps_inner))
            print('Burnin: {}'.format(args.burnin))
            print('Temperature: {}'.format(args.aps_temp))
            with timer():
                d_opt, psi_d = aps_adg_ann(p.d_util, p.a_util, p.prob,
                                           J=args.aps_temp, J_inner=args.aps_temp,
                                           N_aps=args.aps, burnin=args.burnin,
                                           N_inner = args.aps_inner, prec=0.01,
                                           mean=args.aps_mean, info=True)

            psi_d = pd.Series(psi_d)

        else:
            print('Error')

        #a_opt = pd.Series(a_opt, index=d_idx)
        dout = {'d_opt': d_opt, 'psi_d': psi_d}
    #---------------------------------------------------------------------------
    # ARA
    #---------------------------------------------------------------------------
    elif args.set == 'ara':
        print('=' * 30)
        print('ARA')
        print('=' * 30)

        print('ARA iters: {}'.format(args.ara))

        if args.prob:
            p_a = pd.read_pickle('{}'.format(args.prob)).values
            print(p_a)
        else:
            p_a = None

        if args.alg == 'mcmc':
            print('MCMC')
            print('-' * 80)
            print('Iters: {}'.format(args.mcmc))
            with timer():
                d_opt, p_a, psi_da, psi_ad = mcmc_ara(p.d_values, p.a_values,
                                                      p.d_util, p.a_util_f, p.prob,
                                                      p.a_prob_f,
                                                      mcmc_iters=args.mcmc,
                                                      ara_iters=args.ara,
                                                      n_jobs=args.njobs)

        elif args.alg == 'aps':
            print('APS')
            print('-' * 80)
            print('Outer iters: {}'.format(args.aps))
            print('Inner iters: {}'.format(args.aps_inner))
            print('Burnin: {}'.format(args.burnin))
            with timer():
                d_opt, p_a, psi_da = aps_ara(p.d_values, p.a_values,
                                                   p.d_util, p.a_util_f, p.prob,
                                                   p.a_prob_f, N_aps=args.aps,
                                                   J=args.ara,
                                                   burnin=args.burnin, p_d=p_a,
                                                   N_inner=args.aps_inner)
        else:
            print('Error')

        #a_opt = pd.Series(p_a.argmax(axis=1), index=d_idx)
        #psi_d = pd.Series(psi_da.sum(axis=1), index=d_idx)
        #psi_a = pd.DataFrame(psi_ad.mean(axis=2), index=d_idx, columns=a_idx)
        p_a = pd.DataFrame(p_a, index=d_idx, columns=a_idx)
        #dout = {'d_opt': d_opt, 'a_opt': a_opt, 'psi_d': psi_d, 'psi_a': psi_a,
        #        'psi_da': psi_da, 'psi_ad': psi_ad, 'p_a': p_a}
        dout = {'d_opt': d_opt, 'psi_da': psi_da, 'p_a': p_a}
        with pd.option_context('display.max_columns', len(p.a_values)):
            print(p_a)

    else:
        print('Error')
        sys.exit(1)

    print("Optimal Defense:", d_opt)
    #print(a_opt)
    #print(psi_d)
    with pd.option_context('display.max_columns', len(p.a_values)):
        pass
    #print(psi_a)

    fout = '{}/{}_{}_{}.pkl'.format(args.out, args.module, args.alg, args.set)
    with open(fout, "wb") as f:
        pickle.dump(dout, f)
