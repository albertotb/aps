#!/usr/bin/env python
import data.prob_new as p
from mcmc import *
from joblib import Parallel, delayed

disc   = 0.001
times  = 30
iters  = 10000000
inner  = 100000
n_jobs = 16

a_values = np.arange(0, 1, disc)
d_values = np.arange(0, 1, disc)

def find_d_opt():
    d_opt = mcmc_adg(d_values, a_values, p.d_util, p.a_util, p.prob,
                     p.prob, iters=iters, inner_iters=inner, info=False)
    return d_opt

if times == 1:
    d_opt = find_d_opt()
else:
    d_opt = Parallel(n_jobs=n_jobs)(
               delayed(find_d_opt)() for j in range(times)
    )

after_dot = str(disc).split('.', 1)[-1]
np.save(f'./results/sol_{after_dot}.npy', np.array(d_opt))
