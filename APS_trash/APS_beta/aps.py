#!/usr/bin/env python
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import mode


########### APS auxiliar function #################################

def innerAPS(d_given, a_values, a_util, prob, N_inner=1000, burnin_inner=500):
        a_sim = np.zeros(N_inner, dtype = int)
        a_sim[0] = np.random.choice(a_values)
        theta_sim = prob(d_given, a_sim[0])

        for i in range(1,N_inner):
            ## Update a
            a_tilde = 1 if a_sim[i-1]==0 else 0  ## Modify this
            theta_tilde = prob(d_given, a_tilde)

            num = a_util(a_tilde, theta_tilde)
            den = a_util(a_sim[i-1], theta_sim)

            if np.random.uniform() <= num/den:
                a_sim[i] = a_tilde
                theta_sim = theta_tilde
            else:
                a_sim[i] = a_sim[i-1]

        return(mode(a_sim[burnin_inner:])[0][0])



def aps_atk_def(d_values, a_values, d_util, a_util, theta,
N_aps=1000, burnin=500, N_inner = 4000, burnin_inner = 3500):

    ################################################################################
    ### Proposal of new d
    def propose(d_given):
        if d_given == d_values[0]:
            return( np.random.choice([d_values[1], d_values[-1]],
            p=[0.5, 0.5]) )

        if d_given == d_values[-1]:
            return( np.random.choice([d_values[0], d_values[-2]],
            p=[0.5, 0.5]) )

        idx = list(d_values).index(d_given)
        return( np.random.choice([idx + 1, idx - 1],p=[0.5, 0.5]) )

    ## Compute a* for all d
    print("PREPROCESSING...")
    a_star = np.zeros_like(d_values)
    for i, d in enumerate(d_values):
        a_star[d] = innerAPS(d, a_values, a_util, theta, N_inner, burnin_inner)

    # a_star = a_opt
    # print(a_star)
    ## Start d_sim, vector of length N_aps to save in it simulation results
    d_sim = np.zeros(N_aps, dtype = int)
    d_sim[0] = np.random.choice(d_values)
    a_sim = a_star[d_sim[0]]
    theta_sim = theta(d_sim[0], a_sim)

    print("START")

    for i in range(1,N_aps):
        ## Update d
        d_tilde = propose(d_sim[i-1])
        a_tilde = a_star[d_tilde]
        theta_tilde = theta(d_tilde, a_tilde)

        num = d_util(d_tilde, theta_tilde)

        den = d_util(d_sim[i-1], theta_sim)


        if np.random.uniform() <= num/den:
            d_sim[i] = d_tilde
            a_sim = a_tilde
            theta_sim = theta_tilde
        else:
            d_sim[i] = d_sim[i-1]

        # if i%1000 == 0:
        #     print(i)

    return(mode(d_sim[burnin:])[0], a_star)

def aps_ara(d_values, a_values, d_util, a_util_f, theta, a_prob_f,
N_aps=1000, burnin=500, J = 1000, N_inner = 1000, burnin_inner = 500):

    ################################################################################
    ### Proposal of new d
    def propose(d_given):
        if d_given == d_values[0]:
            return( np.random.choice([d_values[1], d_values[-1]],
            p=[0.5, 0.5]) )

        if d_given == d_values[-1]:
            return( np.random.choice([d_values[0], d_values[-2]],
            p=[0.5, 0.5]) )

        idx = list(d_values).index(d_given)
        return( np.random.choice([idx + 1, idx - 1],p=[0.5, 0.5]) )

    ## Compute a* for all d
    print("PREPROCESSING...")
    p_d = np.zeros((len(d_values), len(a_values)), dtype=float)

    for i, d_given in enumerate(d_values):
        modes = np.zeros(J, dtype=int)
        for jj in range(J):
            aUt = a_util_f(n = 1) ### OJO entre generarlos fuera y dentro hay diferencia WHY?????
            pr = a_prob_f(d_given, n = 1)
            modes[jj] = innerAPS( d_given, a_values, aUt[0],  pr[0], N_inner, burnin_inner )
        p_d[i, :] = np.bincount(modes, minlength = 2)/J

    prob_a = lambda d, size=1: bernoulli.rvs(p = p_d[d_values == d, 1], size=size)

    d_sim = np.zeros(N_aps, dtype = int)
    d_sim[0] = np.random.choice(d_values)
    a_sim = prob_a(d_sim[0])
    theta_sim = theta(d_sim[0], a_sim)

    print("START")

    for i in range(1,N_aps):
        ## Update d
        d_tilde = propose(d_sim[i-1])
        a_tilde = prob_a(d_tilde)
        theta_tilde = theta(d_tilde, a_tilde)

        num = d_util(d_tilde, theta_tilde)

        den = d_util(d_sim[i-1], theta_sim)


        if np.random.uniform() <= num/den:
            d_sim[i] = d_tilde
            a_sim = a_tilde
            theta_sim = theta_tilde
        else:
            d_sim[i] = d_sim[i-1]

        # if i%1000 == 0:
        #     print(i)

    return(mode(d_sim[burnin:])[0], p_d)
