#!/usr/bin/env python
import math
import warnings
import numpy as np
import pandas as pd

# Problem parameters
D = 100
c = 50
e = 50
h = 0.1
k = 0.1

# Discretization steps
stepA = 0.01
stepD = 0.01

a_values = np.arange(0, 1, stepA)
d_values = np.arange(0, 1, stepD)


f = lambda d, theta: (1-theta)*D - c*d
g = lambda a, theta: theta*D - e*a

eps = 0.01 # To avoida alpha and beta to be 0
alpha = lambda d,a: a - d + 1 + eps
beta  = lambda d,a: d - a + 1 + eps

prob   = lambda d, a, size=1: np.random.beta( alpha(d,a), beta(d,a), size=size )

n1 = 1 - np.exp(-h*D)
n2 = 1 - np.exp(-h*c)
##
h1 = -1 / (n2-n1)
h2 =  n2 / (n2 - n1)

d_util = lambda d, theta: h1*(1.0 - np.exp( -h * f(d,theta) ) ) + h2
a_util = lambda a, theta: np.exp(-k * g(a,theta) )


if __name__ == '__main__':

    d_values = np.linspace(0,1, num=1000, endpoint=True)
    theta_values = np.linspace(0,1, num=1000, endpoint=True)
    # check all utils are positive
    res = np.zeros((len(d_values), len(theta_values)))
    for i, d in enumerate(d_values):
        for j, theta in enumerate(theta_values):
            res[i, j] = d_util(d, theta)

    print("Max Defender Utility", np.argmax(res))
    print("Min Defender Utility", np.min(res))
