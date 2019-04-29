#!/usr/bin/env python
import math
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import uniform, gamma, beta, binom, norm

CA_MIN = -4.5e6-792*30
CA_MAX = 1.5e6

CD_MIN = 0
CD_MAX = 1.5e6 + 12000

def scale(x, max=None, min=None):
    if not max:
        max = x.max()

    if not min:
        min = x.min()

    return (x - min)/(max - min)

def log_interp1d(xx, yy, kind='linear'):
    logx = np.ma.log10(xx).filled(0)
    logy = np.ma.log10(yy).filled(0)
    #logx = np.log10(xx, where=xx>0)
    #logy = np.log10(yy, where=yy>0)
    lin_interp = interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.where(zz > 0, np.power(10.0,
                                     lin_interp(np.log10(zz, where=zz>0))), 0)
    return log_interp


unif = lambda a, b, size=1: uniform.rvs(a, b-a, size=size)

a_values = np.arange(31)
d_values = np.array([0, 2, 5, 10, 1000])
d_cost = np.array([0, 2400, 3600, 4800, 12000])


cs = log_interp1d(d_values, d_cost)
#cs = lambda d: d_cost[d_values == d]

#d_values = np.array([0, 2, 5, 10, 50, 100, 200, 250, 300, 350, 400, 450,  500, 750, 1000])

d_values = np.arange(100, 200, 5)

def pl(a, d, a_g, scale_g, a_l, scale_l, size=1):
    return (np.where(gamma.rvs(a=a_g, scale=scale_g, size=a*size) - d > 0,
                     gamma.rvs(a=a_l, scale=scale_l, size=a*size), 0)
              .reshape((size, a))
              .sum(axis=1))

def pm(l, alpha, beta):
    return 3e6*l*(alpha+beta)/2

def ct(a, p):
    t = binom.rvs(a, p=p) > 0
    return 2430000 * t

cd = lambda d, l, alpha, beta: cs(d) + pm(l, alpha=alpha, beta=beta)
ud = lambda cd: (1/(math.e-1))*(np.exp(1 - cd/CD_MAX) - 1)

ca = lambda a, l, p, alpha, beta: pm(l, alpha=alpha, beta=beta)-ct(a, p=p)-792*a
def ua(ca, ka=1):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        u = ((ca - CA_MIN)/(CA_MAX - CA_MIN)) ** ka
        if len(w):
            print(ca, ka, u)
    return u

prob   = lambda d, a, size=1: pl(a, d, 100, 1, 10, 1, size=size)
d_util = lambda d, theta: ud(cd(d, theta, 0.0026, 0.00417))
a_util = lambda a, theta: ua(ca(a, theta, 0.002, 0.0026, 0.00417), ka=9)

def a_util_f():
    p        = beta.rvs(2, 998)
    a_param  = unif(0.0021, 0.0031)
    b_param  = unif(0.00367, 0.00467)
    ka       = unif(8, 10)
    return lambda a, theta: ua(ca(a, theta, p, a_param, b_param), ka=ka)

def a_prob_f(d=None):
    a_g     = unif(4.8, 5.6)
    a_l     = unif(3.6, 4.8)
    scale_g = unif(0.8, 1.2)
    scale_l = unif(0.8, 1.2)
    return lambda d, a, size=1: pl(d, a, a_g, scale_g, a_l, scale_l, size=size)

if __name__ == '__main__':

    # check all utils are positive
    res = np.zeros((len(a_values), len(d_values)))
    for i, a in enumerate(a_values):
        for j, d in enumerate(d_values):
            res[i, j] = prob(d, a, size=10000).mean()

    print(pd.DataFrame(res, index=a_values, columns=d_values))


    # l_values = np.linspace(0, gamma.ppf(0.99, a=4, scale=1), 1000)
    #
    # ca_min = ca(a_values.min(), np.repeat(l_values[0], 1000000), 0.002, 0.0026, 0.00417)
    # ca_max = ca(a_values.max(), np.repeat(l_values[-1], 1000000), 0.002, 0.0026, 0.00417)
    #
    # print(ca_min.min())
    # print(ca_max.max())
    #
    # ca_min = ca(a_values.min(), prob(a_values.min(), d_values.max(), size=100000), 0.002, 0.0026, 0.00417)
    # ca_max = ca(a_values.max(), prob(a_values.max(), d_values.min(), size=100000), 0.002, 0.0026, 0.00417)
    #
    # print(ca_min.min())
    # print(ca_max.max())

    #for n in range(1000):
    #    assert (np.array([ (a_util(a, l_values) > 0).all() for a in a_values ]) > 0).all()
    #    assert (np.array([ (d_util(d, l_values) > 0).all() for d in d_values ]) > 0).all()
