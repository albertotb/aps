#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import uniform, beta, bernoulli
import prob1 as p
from mcmc import mcmc_adg, mcmc_ara


if __name__ == '__main__':

    K = 1000
    iters=100000

    e_values = uniform.rvs(scale=2, size=K)

    res = []
    for e in e_values:
        a_util = lambda a, theta: p.ua(p.ca[a, theta], e=e)

        def a_prob(d, a, size=1):
            p1 = beta.rvs(a=p.alpha_values[d], b=p.beta_values[d])
            return bernoulli.rvs(p=p1,
                                 size=size) if a==1 else np.zeros(size, dtype=int)

        d_opt, a_opt, psi_d, psi_a, t = mcmc_adg(p.d_values, p.a_values,
                                                 p.d_util, a_util,
                                                 p.prob, a_prob,
                                                 mcmc_iters=iters)

        res.append({'d_opt': d_opt, 'psi_d': psi_d[d_opt]})

    df = pd.DataFrame.from_dict(res)
    df.to_pickle('prob1_sa.pkl')
