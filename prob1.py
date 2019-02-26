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

E = 1
C = 0.4
UD_MAX = np.exp(cd.max()*C)

# Define utility model
ud = lambda cd, c=1: -np.exp(c*cd)
ua = lambda ca, e=1:  np.exp(e*ca)

prob   = lambda d, a, size=1: bernoulli.rvs(p=p1[d, a], size=size)
d_util = lambda d, theta: ud(cd[d, theta], c=C) + UD_MAX
a_util = lambda a, theta: ua(ca[a, theta], e=E)

def a_util_f():
    e=uniform.rvs(scale=2)
    return lambda a, theta: ua(ca[a, theta], e = e)

def a_prob_f(d=None):
    p1 = beta.rvs(a=alpha_values[d], b=beta_values[d])
    def a_prob(d, a, size=1):
        return bernoulli.rvs(p=p1,
                             size=size) if a==1 else np.zeros(size, dtype=int)
    return a_prob

if __name__ == '__main__':

    # check all utilities are positive
    util_a = ua(data['c_A'], e=E)
    util_d = ud(data['c_D'], c=C) + np.exp(data['c_D'].max()*C)

    assert (util_a >= 0).all()
    assert (util_d >= 0).all()


    cadf = pd.DataFrame(ca, index=pd.Index(a_values, name='a'),
                            columns=pd.Index(theta_values, name='theta'))

    cddf = pd.DataFrame(cd, index=pd.Index(d_values, name='d'),
                            columns=pd.Index(theta_values, name='theta'))

    p1df = pd.DataFrame(p1, index=pd.Index(d_values, name='d'),
                            columns=pd.Index(a_values, name='a'))

    aradf = pd.DataFrame({'alpha': alpha_values, 'beta': beta_values},
                         index=pd.Index(d_values, name='d'))

    print(cadf.to_latex())
    print(cddf.to_latex())
    print(p1df.to_latex())
    print(aradf.to_latex())
