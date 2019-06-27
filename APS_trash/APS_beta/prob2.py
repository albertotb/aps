#!/usr/bin/env python
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import uniform, gamma, beta, binom, norm
from mcmc import mcmc_atk_def, mcmc_ara

N = 1000
M = 100
CA_MAX = 1.5e6
CA_MIN = -4.33e6

unif = lambda a, b, size=1: uniform.rvs(a, b-a, size=size)

a_values = np.arange(31)
d_values = np.array([0, 2, 5, 10, 1000])

#-------------------------------------------------------------------------------
# Attacker-defender game
#-------------------------------------------------------------------------------

d_cost = np.array([0, 2400, 3600, 4800, 12000])
cs = lambda d: d_cost[d_values == d]

def pl(a, d, a_g, scale_g, a_l, scale_l, size=1):
    return (np.where(gamma.rvs(a=a_g, scale=scale_g, size=a*size) - d > 0,
                     gamma.rvs(a=a_l, scale=scale_l, size=a*size), 0)
              .reshape((size, a))
              .sum(axis=1))

def pm(l, alpha, beta):
    return np.minimum(1.5e6, l*unif(alpha, beta))

def ct(a, p):
    t = binom.rvs(a, p=p) > 0
    return norm.rvs(2430000, 400000) * t

cd = lambda d, l, alpha, beta: cs(d) + pm(l, alpha=alpha, beta=beta)
ud = lambda cd: (1/(math.e-1))*(np.exp(1 - cd/7e6) - 1)

ca = lambda a, l, p, alpha, beta: pm(l, alpha=alpha, beta=beta)-ct(a, p=p)-792*a
ua = lambda ca, ka=1: ((ca - CA_MIN)/(CA_MAX - CA_MIN)) ** ka

prob   = lambda d, a, size=1: pl(a, d, 5, 1, 4, 1, size=size)
d_util = lambda d, theta: ud(cd(d, theta, 0.0026, 0.00417))
a_util = lambda a, theta: ua(ca(a, theta, 0.002, 0.0026, 0.00417), ka=9)

# MCMC
d_opt, a_opt = mcmc_atk_def(d_values, a_values, d_util, a_util, prob, n=N)

print(d_opt)
print(a_opt)

#-------------------------------------------------------------------------------
# ARA
#-------------------------------------------------------------------------------

def a_util_f(n=1):
    p_arr     = beta.rvs(2, 998, size=n)
    alpha_arr = unif(0.0021, 0.0031, size=n)
    beta_arr  = unif(0.00367, 0.00467, size=n)
    ka_arr    = unif(8, 10, size=n)

    return [ lambda a, theta: ua(ca(a, theta, p, alpha, beta), ka=ka)
             for p, alpha, beta, ka in zip(p_arr, alpha_arr, beta_arr, ka_arr) ]

def a_prob_f(n=1):
    a_g_arr     = unif(4.8, 5.6, size=n)
    a_l_arr     = unif(3.6, 4.8, size=n)
    scale_g_arr = unif(0.8, 1.2, size=n)
    scale_l_arr = unif(0.8, 1.2, size=n)

    return [ lambda d, a, size=1: pl(d, a, *param, size=size)
             for param in zip(a_g_arr, scale_g_arr, a_l_arr, scale_l_arr) ]

# MCMC
d_opt, p_d = mcmc_ara(d_values, a_values, d_util, a_util_f, prob, a_prob_f, n=N,
                      m=M)

print(d_opt)

df = pd.DataFrame(p_d, index=pd.Index(d_values, name='d'),
                       columns=pd.Index(a_values, name='a'))
df.to_pickle('data2.pkl')

with pd.option_context('display.max_columns', len(a_values)):
    print(df)

# pa = pd.read_csv('data2', header=0, delim_whitespace=True, index_col=['a', 'd'])
# print(pa.unstack())
