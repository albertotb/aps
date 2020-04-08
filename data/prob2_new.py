#!/usr/bin/env python
import math
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Defender's decision cost
cs = np.array([0.   , 3600.        , 4800.        , 5203.29101153,
       5509.81702845, 5759.95860937, 5972.74612902, 6158.77670632,
       6324.60076812, 6474.57005153, 6611.73292272, 6738.31337977,
       6855.98715173, 6966.05038682, 7069.5276606 , 7167.24385155,
       7259.87354382, 7347.97593087, 7432.02006704, 7512.40351745,
       7589.4663844 , 7663.50202754, 7734.7653756 , 7803.47945443,
       7869.84057405, 7934.02249396, 7996.17980049, 8056.45066924,
       8114.95914322, 8171.81702549, 8227.12546256, 8280.97627751,
       8333.45309919, 8384.63232399, 8434.58393927, 8483.37223183,
       8531.05640024, 8577.69108634, 8623.32683855, 8668.01051718])

CA_MIN = -4.5e6-792*30
CA_MAX = 1.5e6

CD_MAX = 1.5e6 + cs[-1]
## Decisions
a_values = np.arange(31)
d_values = np.arange(0, 200, 5)



## Probabilities
def pl(a, d, a_g, b_g, a_l, b_l, size=1):
    return (np.where(np.random.gamma(a_g, 1/b_g, size=a*size) - d > 0,
                     np.random.gamma(a_l, 1/b_l, size=a*size), 0.0)
              .reshape((size, a))
              .sum(axis=1))

def pm(l, alpha, beta):
    return np.minimum(1.5e6, 3e6*l*np.random.uniform(alpha, beta, size=len(l)))
    #return 3e6*l*(alpha+beta)/2

def ct(a, p, size):
    t = np.random.binomial(a, p, size=size) > 0
    return np.random.normal(2430000, 400000, size=size) * t
    #return 2430000 * t


# Attacker probability
def prob_a(a, d, params, size=10):
    a_g   = params["a_g"]
    b_g   = params["b_g"]
    a_l   = params["a_l"]
    b_l   = params["b_l"]
    alpha = params["alpha"]
    beta  = params["beta"]
    p     = params["p"]

    sample = np.zeros([size, 3])
    longitudes = pl(a, d, a_g, b_g, a_l, b_l, size=size)
    m_shares   = pm(longitudes, alpha, beta)
    det_costs  = ct(a, p, size=size)

    sample[:,0] = longitudes
    sample[:,1] = m_shares
    sample[:,2] = det_costs

    return sample

# Defender probability
def prob_d(a, d,  size=1):
    sample = np.zeros([size, 2])
    longitudes = pl(a, d, a_g=5, b_g=1, a_l=4, b_l=1, size=size)
    m_shares   = pm(longitudes, 0.0026, 0.00417)
    sample[:,0] = longitudes
    sample[:,1] = m_shares
    return sample


# Attacker's Utility
def ua(a, theta, params):
    ka = params["ka"]
    ca = theta[:,1] - theta[:,2] - 792*a
    return ( (ca - CA_MIN)/(CA_MAX - CA_MIN) )**ka


# Defender's Utility
def ud(d, theta):
    cd = theta[:,1] + cs[np.where(d_values == d)[0][0]]
    return (1/(np.exp(1)-1))*(np.exp(1 - cd/CD_MAX) - 1)


def sample_params():
    a_g    = np.random.uniform(4.8, 5.6)
    b_g    = np.random.uniform(0.8, 1.2)
    a_l    = np.random.uniform(3.6, 4.8)
    b_l    = np.random.uniform(0.8, 1.2)
    alpha  = np.random.uniform(0.0021, 0.0031)
    beta   = np.random.uniform(0.00367, 0.00467)
    p      = np.random.beta(2.0, 998.0)
    ka     = np.random.uniform(8.0, 10.0)

    params = { "a_g"   : a_g,
               "b_g"   : b_g,
               "a_l"   : a_l,
               "b_l"   : b_l,
               "alpha" : alpha,
               "beta"  : beta,
               "p"     : p,
               "ka"    : ka
               }

    return params
