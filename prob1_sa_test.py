#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import uniform, beta, bernoulli
import prob1 as p
from mcmc import mcmc_adg, mcmc_ara



K = 1000
iters=100000


d_opt, a_opt, psi_d, psi_a, t = mcmc_adg(p.d_values, p.a_values,
                                                 p.d_util, p.a_util,
                                                 p.prob, p.prob,
                                                 mcmc_iters=iters)

print(d_opt, psi_d)
