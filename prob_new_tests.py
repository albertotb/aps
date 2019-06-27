#!/usr/bin/env python
import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mode
from timeit import default_timer
from importlib import import_module
from contextlib import contextmanager

case = "discrete"

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

if case == "discrete":
    ##
    a_values = np.arange(0, 1, 0.01)
    d_given = 0.9
    mcmc_iters = 10000
    N_inner = 100000
    ################################################################################
    # MC for attacker. Discrete case
    ################################################################################
    psi_a = np.zeros(len(a_values), dtype=float)

    start = default_timer()
    ##
    for j, a in enumerate(a_values):
        theta_a = prob(d_given, a, size=mcmc_iters)
        psi_a[j] = a_util(a, theta_a).mean()
    ##
    end = default_timer()
    print('Elapsed MC time: ', end-start)
    a_opt = a_values[psi_a.argmax()]
    print('Optimal MC attack for given defense', a_opt)

    ################################################################################
    # APS for attacker. Discrete case
    ################################################################################
    # def propose(x_given, x_values, prop = 0.1):
    #     tochoose = int(len(x_values)*prop)
    #     if x_given == x_values[0]:
    #         return( np.random.choice([x_values[1], x_values[-1]],
    #         p=[0.5, 0.5]) )
    #
    #     if x_given == x_values[-1]:
    #         return( np.random.choice([x_values[0], x_values[-2]],
    #         p=[0.5, 0.5]) )
    #
    #     coin = np.random.choice([0,1])
    #     idx = list(x_values).index(x_given)
    #     if coin == 0:
    #         return( np.random.choice(a_values[idx + 1 : idx + 1 + tochoose]) )
    #     else:
    #         return( np.random.choice(a_values[idx + 1 : idx + 1 + tochoose]) )

    def propose(x_given, x_values):
        if x_given == x_values[0]:
            return( np.random.choice([x_values[1], x_values[-1]],
            p=[0.5, 0.5]) )

        if x_given == x_values[-1]:
            return( np.random.choice([x_values[0], x_values[-2]],
            p=[0.5, 0.5]) )

        idx = list(x_values).index(x_given)
        return( np.random.choice([x_values[idx+1], x_values[idx-1]],
        p=[0.5, 0.5]) )

    a_sim = np.zeros(N_inner, dtype = float)
    a_sim[0] = 0.1
    theta_sim = prob(d_given, a_sim[0])

    start = default_timer()
    for i in range(1,N_inner):
        ## Update a
        a_tilde = propose(a_sim[i-1], a_values)
        theta_tilde = prob(d_given, a_tilde)

        num = a_util(a_tilde, theta_tilde)
        den = a_util(a_sim[i-1], theta_sim)

        if np.random.uniform() <= num/den:
            a_sim[i] = a_tilde
            theta_sim = theta_tilde
        else:
            a_sim[i] = a_sim[i-1]
    end = default_timer()
    ##
    print('Elapsed APS time: ', end-start)
    ##
    burnin = 0.5
    a_dist = a_sim[int(burnin*N_inner):]
    a_opt = mode(a_dist)[0][0]
    print('Optimal APS attack for given defense', a_opt)



################################################################################
## APS for attacker, continuous case
################################################################################
if case == "continuous":
    d_given = 0.01
    N_inner = 100000

    a_sim = np.zeros(N_inner, dtype = float)
    a_sim[0] = 0.1
    theta_sim = prob(d_given, a_sim[0])

    a_sim[6442]
    ##
    def beta_params(mu, var):
        alpha = ( (1-mu)/var - 1/mu ) * mu**2
        beta = ( 1/mu - 1 )*alpha
        return alpha, beta

    def propose(mu, var):
        a, b = beta_params(mu, var)
        prop = np.random.beta(a, b)
        return prop

    var = 0.001
    for i in range(1,N_inner):
        print(i)
        ## Update a
        a_tilde = propose(a_sim[i-1], var)
        theta_tilde = prob(d_given, a_tilde)

        num = a_util(a_tilde, theta_tilde)
        den = a_util(a_sim[i-1], theta_sim)

        if np.random.uniform() <= num/den:
            a_sim[i] = a_tilde
            theta_sim = theta_tilde
        else:
            a_sim[i] = a_sim[i-1]
    np.max(a_sim)
    burnin = 0.25
    a_dist = a_sim[int(burnin*N_inner):]
    mode(a_dist)[0][0]
