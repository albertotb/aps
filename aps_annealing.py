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
def propose_att(current):
    return( np.random.uniform(0,1) )

def propose(current):
    prop = current + np.random.normal(0, 0.01)
    return prop

def propose_init(current):
    prop = current + np.random.normal(0,0.1)
    return prop
    

############################################################################
### Inner APS ##############################################################
############################################################################

def innerAPS(J, d_given, a_util, theta, N_inner=1000, mean=False, burnin=0.1, prec = 0.01,
    info=False):
    #
    a_sim = np.zeros(N_inner, dtype = float)
    a_sim[0] = 0.5
    theta_sim = theta(d_given, a_sim[0], size=J)

    for i in range(1,N_inner):
        ## Update a
        a_tilde = propose_att(a_sim[i-1])
        # a_tilde = propose(a_sim[i-1]) if i > int(N_inner*0.5) else propose_init(a_sim[i-1])
        theta_tilde = theta(d_given, a_tilde, size=J)
        num = a_util(a_tilde, theta_tilde)
        den = a_util(a_sim[i-1], theta_sim)
        if np.random.uniform() <= np.prod(num/den):
            a_sim[i] = a_tilde
            theta_sim = theta_tilde
        else:
            a_sim[i] = a_sim[i-1]

    a_dist = a_sim[int(burnin*N_inner):]
    if mean:
        a_opt = np.mean(a_dist)
    else:
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

def aps_adg_ann(J, J_inner, d_util, a_util, theta, N_aps=1000,
                burnin=0.75, N_inner = 4000, prec=0.01, mean=False, info=True):

    d_sim = np.zeros(N_aps, dtype = float)
    d_sim[0] = 0.5
    a_sim = innerAPS(J_inner, d_sim[0], a_util, theta, N_inner=N_inner, mean=mean)
    theta_sim = theta(d_sim[0], a_sim, size=J)

    for i in range(1,N_aps):
        ## Update d
        #d_tilde = propose(d_sim[i-1])
        d_tilde = propose(d_sim[i-1]) if i > int(N_inner*0.1) else propose_init(d_sim[i-1])
        a_tilde = innerAPS(J_inner, d_tilde, a_util, theta, N_inner=N_inner, mean=mean)
        theta_tilde = theta(d_tilde, a_tilde, size=J)

        num = d_util(d_tilde, theta_tilde)

        den = d_util(d_sim[i-1], theta_sim)

        if np.random.uniform() <= np.prod(num/den):
            d_sim[i] = d_tilde
            a_sim = a_tilde
            theta_sim = theta_tilde
        else:
            d_sim[i] = d_sim[i-1]

    d_dist = d_sim[int(burnin*N_aps):]
    ###
    ###
    if mean:
        d_opt = np.mean(d_dist)
    else:
        count, bins = np.histogram(d_dist, bins = int(1.0/prec) )
        d_opt = ( bins[count.argmax()] + bins[count.argmax()+1] ) / 2
    ###
    ###
    if info:
        d_d = pd.Series(d_dist)
        return d_opt, d_d
    ###
    return d_opt
