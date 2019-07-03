#!/usr/bin/env python
import math
import warnings
import numpy as np
import pandas as pd
import math
from scipy.stats import mode
from timeit import default_timer
import sys
sys.path.append('.')
from prob_new import *

### Proposal Distribution
def propose():
    return( np.random.uniform(0,1) )

############################################################################
### Inner APS ##############################################################
############################################################################

def innerAPS(d_given, a_util, theta, N_inner=1000, burnin=0.75, prec = 0.01,
    info=False):
    #
    a_sim = np.zeros(N_inner, dtype = float)
    a_sim[0] = 0.5
    theta_sim = theta(d_given, a_sim[0])

    for i in range(1,N_inner):
        ## Update a
        a_tilde = propose()
        theta_tilde = theta(d_given, a_tilde)
        num = a_util(a_tilde, theta_tilde)
        den = a_util(a_sim[i-1], theta_sim)
        if np.random.uniform() <= num/den:
            a_sim[i] = a_tilde
            theta_sim = theta_tilde
        else:
            a_sim[i] = a_sim[i-1]

    a_dist = a_sim[int(burnin*N_inner):]
    count, bins = np.histogram(a_dist, bins = int(1.0/prec) )
    a_opt = ( bins[count.argmax()] + bins[count.argmax()+1] ) / 2
    if info:
        a_d = pd.Series(a_dist)
        return a_opt, a_d
    ##
    return a_opt

############################################################################
### Outer APS ##############################################################
############################################################################

def aps_adg(d_util, a_util, theta, N_aps=1000,
                burnin=0.75, N_inner = 4000, prec=0.01, info=False):

    d_sim = np.zeros(N_aps, dtype = float)
    d_sim[0] = 0.5
    a_sim = innerAPS(d_sim[0], a_util, theta, N_inner=N_inner)
    theta_sim = theta(d_sim[0], a_sim)

    for i in range(1,N_aps):
        ## Update d
        d_tilde = propose()
        a_tilde = innerAPS(d_tilde, a_util, theta, N_inner=N_inner)
        theta_tilde = theta(d_tilde, a_tilde)

        num = d_util(d_tilde, theta_tilde)

        den = d_util(d_sim[i-1], theta_sim)

        if np.random.uniform() <= num/den:
            d_sim[i] = d_tilde
            a_sim = a_tilde
            theta_sim = theta_tilde
        else:
            d_sim[i] = d_sim[i-1]

    d_dist = d_sim[int(burnin*N_aps):]
    count, bins = np.histogram(d_dist, bins = int(1.0/prec) )
    d_opt = ( bins[count.argmax()] + bins[count.argmax()+1] ) / 2
    if info:
        d_d = pd.Series(d_dist)
        return d_opt, d_d
    return(d_opt, d_dist)
