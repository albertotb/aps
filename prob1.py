#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, uniform, beta

data = pd.read_table('data1', header=0, delim_whitespace=True)

data = data.sort_values(by = ['d','a','theta'])

# We assume that d, a and theta take values from 0 to
# len({d_values, a_values, theta_values})-1
d_values = data['d'].unique()
a_values = data['a'].unique()
theta_values = data['theta'].unique()

alpha_values = data['alpha'].dropna().values
beta_values = data['beta'].dropna().values

# Important to set the index in this order and sort it before the reshape
data = data.set_index(['d', 'a', 'theta'])

ca = (data['c_A'].groupby(['a', 'theta'])
                 .first() # this cost should be independent of d
                 .values.reshape(len(a_values), len(theta_values)))

cd = (data['c_D'].groupby(['d', 'theta'])
                 .first() # this cost should be independent of a
                 .values.reshape(len(d_values), len(theta_values)))

p1 = (data.loc[pd.IndexSlice[:, :, 1], 'p']
          .values.reshape(len(d_values), len(a_values)))

UD_MAX = np.exp(cd.max()*0.4)

# Define utility model
ud = lambda cd, c=1: -np.exp(c*cd)
ua = lambda ca, e=1:  np.exp(e*ca)

prob   = lambda d, a, size=1: bernoulli.rvs(p=p1[d, a], size=size)
d_util = lambda d, theta: ud(cd[d, theta], c=0.4) + UD_MAX
a_util = lambda a, theta: ua(ca[a, theta], e=10)

def a_util_f():
    return lambda a, theta: ua(ca[a, theta], e=uniform.rvs(scale=20))

def a_prob_f():
    def a_prob(d, a, size=1):
        return bernoulli.rvs(p=beta.rvs(a=alpha_values[d], b=beta_values[d]),
                             size=size) if a==1 else np.zeros(size, dtype=int)
    return a_prob
