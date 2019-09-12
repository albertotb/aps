#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import pickle as pkl
from prob_new import *
from aps_annealing import aps_adg_ann
from mcmc import mcmc_adg
from timeit import repeat

sys.path.append('.')

if __name__ == '__main__':

    # TODO: read the optimal number of iterations from the CSVs
    #
    #if len(sys.argv) != 3:
    #    print("Expected 3 arguments. Given {}".format(len(sys.argv)))
    #    sys.exit(1)

    #print(pd.read_csv(sys.argv[1]))
    #print(pd.read_csv(sys.argv[2]))

    if len(sys.argv) == 4:
        step = float(sys.argv[1])
        outer_iters = int(sys.argv[2])
        inner_iters = int(sys.argv[3])
        fstr = f'''mcmc_adg(d_values, a_values, d_util, a_util, prob, prob,
                            mcmc_iters={outer_iters},
                            inner_mcmc_iters={inner_iters}, info=True)'''

    elif len(sys.argv) == 6:
        step = float(sys.argv[1])
        outer_iters = int(sys.argv[2])
        inner_iters = int(sys.argv[3])
        outer_power = int(sys.argv[4])
        inner_power = int(sys.argv[5])
        fstr = f'''aps_adg_ann(d_util, a_util, prob, J={outer_power},
                               J_inner={inner_power}, N_aps={outer_iters},
                               N_inner={inner_iters}, prec={step}, mean=True,
                               burnin=0.2, info=False)'''
    else:
        print('usage: {} STEP OUTER_ITERS INNER_ITERS'.format(sys.argv[0]))
        print('       {} STEP OUTER_ITERS INNER_ITERS OUTER_POWER INNER_POWER'.format(sys.argv[0]))
        sys.exit(1)

    n = 50
    r = 3

    a_values = np.arange(0, 1, step)
    d_values = np.arange(0, 1, step)

    t = repeat(fstr, number=n, repeat=r, globals=globals())

    print(sys.argv[1:])
    print(t)
