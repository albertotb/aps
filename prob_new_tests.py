#!/usr/bin/env python
import math
import warnings
import numpy as np
import pandas as pd
import math
from scipy.stats import mode
from timeit import default_timer
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from prob_new import *

case = "continuous"
plot = False
aps = True
# d_given = 0.9

if case == "discrete":
    if len(sys.argv) > 1:
        d_given = float(sys.argv[1])
    ##
    # a_values = np.arange(0, 1, 0.01)
    a_optimal = 0.99
    mcmc_iters = 100000
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

    if plot:
        df = pd.DataFrame({"Attacks":a_values, "Exp-Ut":psi_a})
        df.plot.bar(x='Attacks', y='Exp-Ut')
        plt.show()

    if aps:
        N_inner = 10000
        burnin = 0.5
        ################################################################################
        # APS for attacker. Discrete case
        ################################################################################
        def propose(x_given, x_values, prop=0.1):
            tochoose = int(len(x_values)*prop)
            coin = np.random.choice([0,1])
            idx = list(x_values).index(x_given)
            if coin == 0:
                start = idx+1
                end = start + tochoose
                if end >= len(x_values):
                    candidates = np.concatenate((x_values[start:], a_values[:end-len(x_values)]))
                else:
                    candidates = x_values[start:end]
            else:
                start = idx
                end = start-tochoose
                if end < 0:
                    candidates = np.concatenate((x_values[:start][::-1], x_values[end:][::-1]))
                else:
                    candidates = x_values[end:start][::-1]
            return( np.random.choice(candidates) )


        #def propose(x_given, x_values):
        #    if x_given == x_values[0]:
        #        return( np.random.choice([x_values[1], x_values[-1]],
        #        p=[0.5, 0.5]) )

        #    if x_given == x_values[-1]:
        #        return( np.random.choice([x_values[0], x_values[-2]],
        #        p=[0.5, 0.5]) )

        #    idx = list(x_values).index(x_given)
        #    return( np.random.choice([x_values[idx+1], x_values[idx-1]],
        #    p=[0.5, 0.5]) )

        a_sim = np.zeros(N_inner, dtype = float)
        a_sim[0] = np.random.choice(a_values)
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

            if i > burnin*N_inner:
                a_dist = a_sim[int(burnin*N_inner):i]
                a_opt = mode(a_dist)[0][0]
                ##
                if math.isclose(a_opt, a_optimal):
                    break
        end = default_timer()
        ##
        print('Elapsed APS time: ', end-start)
        ##
        #a_opt = mode(a_dist)[0][0]
        print('Optimal APS attack for given defense', a_opt)

################################################################################
## APS for attacker, continuous case
################################################################################
if case == "continuous":
    if len(sys.argv) > 1:
        d_given = float(sys.argv[1])
    ##
    ############################################################################
    ### Monte Carlo ############################################################
    mcmc_iters = 100000

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
    ##
    ############################################################################
    ### APS ############################################################
    N_inner = 100000
    prec = 0.01
    #
    a_sim = np.zeros(N_inner, dtype = float)
    a_sim[0] = np.random.uniform(0,1)
    theta_sim = prob(d_given, a_sim[0])
    ##
    # def beta_params(mu, var):
    #     alpha = ( (1-mu)/var - 1/mu ) * mu**2
    #     beta = ( 1/mu - 1 )*alpha
    #     return alpha, beta
    #
    # def propose(mu, var):
    #     a, b = beta_params(mu, var)
    #     prop = np.random.beta(a, b)
    #     return prop
    def propose():
        return( np.random.uniform(0,1) )

    # var = 0.001
    start = default_timer()
    for i in range(1,N_inner):
        ## Update a
        a_tilde = propose()#propose(a_sim[i-1], var)
        theta_tilde = prob(d_given, a_tilde)

        num = a_util(a_tilde, theta_tilde)
        den = a_util(a_sim[i-1], theta_sim)

        if np.random.uniform() <= num/den:
            a_sim[i] = a_tilde
            theta_sim = theta_tilde
        else:
            a_sim[i] = a_sim[i-1]
    end = default_timer()
    print('Elapsed APS time: ', end-start)
    ##
    burnin = 0.25
    a_dist = a_sim[int(burnin*N_inner):]
    # a_opt = mode(a_dist)[0][0]
    #a_dist = np.array([1,1,2,2,3,4,5,3.5])
    count, bins = np.histogram(a_dist, bins = int(1.0/prec) )
    a_opt = ( bins[count.argmax()] + bins[count.argmax()+1] ) / 2
    #a_opt = mode(a_dist)[0][0]
    print('Optimal APS attack for given defense', a_opt)

    if plot:
        a_d = pd.Series(a_dist)
        a_d.hist(bins = int(1.0/prec))
        plt.show()
