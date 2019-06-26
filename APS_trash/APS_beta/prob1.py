#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, uniform, beta
from mcmc import mcmc_atk_def, mcmc_ara
from aps import aps_atk_def, aps_ara

N = 10000
M = 1000

data = pd.read_table('data1', header=0, delim_whitespace=True)
print(data)
print(data.shape)
print(data.dtypes)

data = data.sort_values(by=['d', 'a', 'theta'])

d_values = data['d'].unique()
a_values = data['a'].unique()
theta_values = data['theta'].unique()

# Important to sort the index before the reshape
# We also assume theta only takes values {0, 1})
#data = data.set_index(['d', 'a', 'theta']).sort_index()

dd = { k: data[k].values.reshape(len(d_values), len(a_values), len(theta_values))
       for k in ('p', 'c_D', 'c_A') }

alpha_arr = data['alpha'].dropna().values
beta_arr = data['beta'].dropna().values

# Define utility model
ud = lambda Cd, c=0.4: np.exp(-c*Cd)
ua = lambda Ca, e=0.4: np.exp(e*Ca)

#-------------------------------------------------------------------------------
# Attacker-defender game
#-------------------------------------------------------------------------------

# MCMC
prob   = lambda d, a, size=1: bernoulli.rvs(p = dd['p'][d_values == d,
                                            a_values == a, 1], size=size)
d_util = lambda d, theta: ud(dd['c_D'][d_values == d, 0, theta])
a_util = lambda a, theta: ua(dd['c_A'][0, a_values == a, theta])

d_opt, a_opt = mcmc_atk_def(d_values, a_values, d_util, a_util, prob, n=N)

print(d_opt)
print(a_opt)

# APS
d_opt, a_opt = aps_atk_def(d_values, a_values, d_util, a_util, prob,
               N_aps=100000, burnin=75000, N_inner=15000, burnin_inner=10000)

print(d_opt)
print(a_opt)

#-------------------------------------------------------------------------------
# ARA
#-------------------------------------------------------------------------------

# def a_prob_f(d, n=1):
#     p1_arr = beta.rvs(a=alpha_arr[d_values == d], b=beta_arr[d_values == d], size = n)
#     def a_prob(d, a, size=1):
#         if a == 0:
#             return np.zeros(size, dtype=int)
#         else:
#             return bernoulli.rvs(p=p1, size=size)
#
#     return [ lambda a, d, size=1: a_prob(a, d, size=size) for p1 in p1_arr ]
def a_prob_f(d, n=1):
    p1_arr = beta.rvs(a=alpha_arr[d_values == d], b=beta_arr[d_values == d], size = n)
    return ( [lambda d, a, size=1: bernoulli.rvs(p=p1, size=size) if a==1 else np.zeros(size, dtype=int) for p1 in p1_arr] )


def a_util_f(n=1):
    return [ lambda a, theta: ua(dd['c_A'][0, a_values == a, theta], e=e)
             for e in uniform.rvs(scale=20, size=n) ]

# MCMC
d_opt, p_d = mcmc_ara(d_values, a_values, d_util, a_util_f, prob, a_prob_f, n=N,
                      m=M)

print(d_opt)
with pd.option_context('display.max_colwidth', -1):
    print(pd.DataFrame(p_d, index=d_values, columns=a_values))


# APS
d_opt, p_d = aps_ara(d_values, a_values, d_util, a_util_f, prob, a_prob_f,
N_aps=100000, burnin=75000, J = 10000, N_inner = 10000, burnin_inner = 7500)

print(d_opt)
with pd.option_context('display.max_colwidth', -1):
    print(pd.DataFrame(p_d, index=d_values, columns=a_values))
