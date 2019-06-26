#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
from importlib import import_module
from mcmc import mcmc_atk_def, mcmc_ara
from aps import aps_atk_def, aps_ara

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

    # np.random.seed(1234)

    #---------------------------------------------------------------------------
    # Attacker-defender game
    #---------------------------------------------------------------------------
    print('Game theory')
    print('-' * 80)

    # MCMC
    d_opt, a_opt = mcmc_atk_def(p.d_values, p.a_values, p.d_util, p.a_util,
                                p.prob, n=args.mcmc)

    print('MCMC')
    print(d_opt)
    print(a_opt)

    # APS
    d_opt, a_opt = aps_atk_def(p.d_values, p.a_values, p.d_util, p.a_util,
                               p.prob, N_aps=args.aps, N_inner=args.aps_inner,
                               burnin=args.burnin)

    print('APS')
    print(d_opt)
    print(a_opt)

    #---------------------------------------------------------------------------
    # ARA
    #---------------------------------------------------------------------------
    print('\nARA')
    print('-' * 80)

    # MCMC
    d_opt, p_d = mcmc_ara(p.d_values, p.a_values, p.d_util, p.a_util_f, p.prob,
                          p.a_prob_f, n=args.mcmc, m=args.ara)

    df1 = pd.DataFrame(p_d, index=pd.Index(p.d_values, name='d'),
                            columns=pd.Index(p.a_values, name='a'))
    df1.to_pickle('{}_mcmc_pa.pkl'.format(args.module))

    print('MCMC')
    print(d_opt)
    with pd.option_context('display.max_columns', len(p.a_values)):
        print(df1)

    # APS
    d_opt, p_d = aps_ara(p.d_values, p.a_values, p.d_util, p.a_util_f, p.prob,
                         p.a_prob_f, N_aps=args.aps, N_inner=args.aps_inner,
                         burnin=args.burnin, J=args.ara)

    df2 = pd.DataFrame(p_d, index=pd.Index(p.d_values, name='d'),
                            columns=pd.Index(p.a_values, name='a'))
    df2.to_pickle('{}_aps_pa.pkl'.format(args.module))

    print('APS')
    print(d_opt)
    with pd.option_context('display.max_columns', len(p.a_values)):
        print(df2)
