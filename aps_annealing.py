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

def innerAPS(J, d_given, a_util, theta, N_inner=1000, burnin=0.75, prec = 0.01,
    info=False):
    #
    a_sim = np.zeros(N_inner, dtype = float)
    a_sim[0] = 0.5
    theta_sim = theta(d_given, a_sim[0], size=J)

    for i in range(1,N_inner):
        ## Update a
        a_tilde = propose()
        theta_tilde = theta(d_given, a_tilde, size=J)
        num = a_util(a_tilde, theta_tilde)
        den = a_util(a_sim[i-1], theta_sim)
        if np.random.uniform() <= np.prod(num/den):
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
