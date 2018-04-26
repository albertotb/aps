#!/usr/bin/env python
import math
import numpy as np
import pandas as pd
from scipy.stats import uniform, gamma, beta, binom, norm

CA_MAX = 1.5e6
CA_MIN = -4.33e6

unif = lambda a, b, size=1: uniform.rvs(a, b-a, size=size)

a_values = np.arange(31)
d_values = np.array([0, 2, 5, 10, 1000])

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

def a_util_f():
    p        = beta.rvs(2, 998)
    a_param  = unif(0.0021, 0.0031)
    b_param  = unif(0.00367, 0.00467)
    ka       = unif(8, 10)
    return lambda a, theta: ua(ca(a, theta, p, a_param, b_param), ka=ka)

def a_prob_f():
    a_g     = unif(4.8, 5.6)
    a_l     = unif(3.6, 4.8)
    scale_g = unif(0.8, 1.2)
    scale_l = unif(0.8, 1.2)
    return lambda d, a, size=1: pl(d, a, a_g, scale_g, a_l, scale_l, size=size)

if __name__ == '__main__':

    # check all utils all positive
    gmin, gmax = gamma.ppf([0.01, 0.99], a=4, scale=1)
    l_values = np.linspace(gmin, gmax, 1000)

    for n in range(1000):
        assert (np.array([ (a_util(a, l_values) > 0).all() for a in a_values ]) > 0).all()
        assert (np.array([ (d_util(d, l_values) > 0).all() for d in d_values ]) > 0).all()
