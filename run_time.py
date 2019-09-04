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

    n = 50
    r = 3

    t1 = repeat('''mcmc_adg(d_values, a_values, d_util, a_util, prob, prob,
                            mcmc_iters=1000, inner_mcmc_iters=1000, info=True)
                ''', number=n, repeat=r, globals=globals())

    t2 = repeat('''mcmc_adg(d_values, a_values, d_util, a_util, prob, prob,
                            mcmc_iters=111000, inner_mcmc_iters=111000, info=True)
                ''', number=n, repeat=r, globals=globals())

    t3 = repeat('''aps_adg_ann(40, 40, d_util, a_util, prob, N_aps=50,
                              burnin=0.1, N_inner=50, prec=0.1, mean=True,
                              info=False)
                ''', number=n, repeat=r, globals=globals())

    t4 = repeat('''aps_adg_ann(100000, 1000, d_util, a_util, prob, N_aps=1000,
                              burnin=0.1, N_inner=100, prec=0.01, mean=True,
                              info=False)
                ''', number=n, repeat=r, globals=globals())

    print(t1)
    print(t2)
    print(t3)
    print(t4)

    with open('./results/times.pkl', 'wb') as fp:
        pkl.dump((t1, t2, t3, t4), fp)
