#!/usr/bin/env python
import math
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Problem parameters
D = 1000
c = 10
e = 50
h = 0.1
k = 0.2

# Discretization steps
stepA = 0.01
stepD = 0.01

a_values = np.arange(0, 1, stepA)
d_values = np.arange(0, 1, stepD)


f = lambda d, theta: (1-theta)*D - c*d
g = lambda a, theta: theta*D - e*a

eps = 0.01 # To avoida alpha and beta to be 0
alpha = lambda d,a: (a-d)**2 + eps
beta  = lambda d,a: (d-a)**2 + eps

prob   = lambda d, a, size=1: np.random.beta( alpha(d,a), beta(d,a), size=size )

d_util = lambda d, theta: 1.0 - np.exp( -h * f(d,theta) )
a_util = lambda a, theta: np.exp(-k * g(a,theta) )


if __name__ == '__main__':

    # check all utils are positive
    res = np.zeros((len(a_values), len(d_values)))
    for i, a in enumerate(a_values):
        for j, d in enumerate(d_values):
            res[i, j] = prob(d, a, size=10000).mean()

    print(pd.DataFrame(res, index=a_values, columns=d_values))


