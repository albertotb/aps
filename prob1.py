#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, uniform, beta

data = pd.read_table('data1', header=0, delim_whitespace=True)

# Important to sort the columns in this order before the reshape
data = data.sort_values(by=['d', 'a', 'theta'])

d_values = data['d'].unique()
a_values = data['a'].unique()
theta_values = data['theta'].unique()

# We assume that d, a and theta take values from 0 to
# len({d_values, a_values, theta_values})-1

dd = { k: data[k].values.reshape(len(d_values), len(a_values), len(theta_values))
       for k in ('p', 'c_D', 'c_A') }

alpha_values = data['alpha'].dropna().values
beta_values = data['beta'].dropna().values

# Define utility model
ud = lambda Cd, c=0.4: np.exp(-c*Cd)
ua = lambda Ca, e=0.4: np.exp(e*Ca)

prob   = lambda d, a, size=1: bernoulli.rvs(p = dd['p'][d, a, 1], size=size)
d_util = lambda d, theta: ud(dd['c_D'][d, 0, theta])
a_util = lambda a, theta: ua(dd['c_A'][0, a, theta])

def a_util_f():
    e = uniform.rvs(scale=20)
    return lambda a, theta: ua(dd['c_A'][0, a, theta], e=e)

def a_prob_f(d):
    p1 = beta.rvs(a=alpha_values[d], b=beta_values[d])
    return lambda d, a, size=1: (bernoulli.rvs(p=p1, size=size) if a == 1
                                 else np.zeros(size, dtype=int))
