#!/usr/bin/env python
import math
import warnings
import numpy as np
import pandas as pd

# Problem parameters
D = 1000
c = 50
e = 10
h = 1.0
k = 0.01

# Discretization steps
stepA = 0.1
stepD = 0.01

a_values = np.arange(0, 10, stepA)
d_values = np.arange(0, 1, stepD)


f = lambda d, theta: (1-theta)*D - c*d
g = lambda a, theta: theta*D - e*a

eps = 0.01 # To avoid alpha and beta to be 0
alpha = lambda d,a: np.exp(a - d)  #+ eps
beta  = lambda d,a: np.exp(d - a)  #+ eps

prob = lambda d, a, size=1: np.random.beta( alpha(d,a), beta(d,a), size=size )

c1 = 1 - np.exp(-h*D)
c2 = 1 - np.exp(h*c)

e1 = np.exp(-k*D)
e2 = np.exp(k*e)

##
h1 =   1 / (c1 - c2)
h2 = -c2 / (c1 - c2)

k1 = 1 / (e1 - e2)
k2 = -e2 / (e1 - e2)

d_util = lambda d, theta: h1*(1.0 - np.exp( -h * f(d,theta) ) ) + h2
a_util = lambda a, theta: k1*(np.exp(-k * g(a,theta) )) + k2


if __name__ == '__main__':

    a_values = np.linspace(0,1, num=1000, endpoint=True)
    d_values = np.linspace(0,1, num=1000, endpoint=True)
    theta_values = np.linspace(0,1, num=1000, endpoint=True)

    # check all utils are positive
    a_res = np.zeros((len(theta_values), len(a_values)))
    d_res = np.zeros((len(theta_values), len(d_values)))
    for i, theta in enumerate(theta_values):
        for j, d in enumerate(d_values):
            d_res[i, j] = d_util(d, theta)

        for j, a in enumerate(a_values):
            a_res[i, j] = a_util(a, theta)

    print("Max Defender Utility", np.max(d_res))
    print("Min Defender Utility", np.min(d_res))

    print("Max Attacker Utility", np.max(a_res))
    print("Min Attacker Utility", np.min(a_res))
