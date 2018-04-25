#!/usr/bin/env python
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import mode


########### APS auxiliar function #################################
### Proposal of new d
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

def innerAPS(d_given, a_values, a_util, theta, N_inner=1000, burnin=0.75):
    """ Computes the attacker's solution in an attacker-defender game using APS

        Parameters
        ----------
        d_given : Integer or string
            Decision of the defender
        a_values : array-like
            Strategies for the attacker
        a_util : callable
            Computes the utility of the attacker

                ``a_util(a, theta) -> array, shape (same as theta)``

            where theta is a 1-D array and a is the decision of the attacker
        theta : callable
            Draws samples of theta from the distribution p(theta | d, a)

                ``theta(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions for the defender and the attacker
        N_inner : int, optional (default=1000)
            Number of iterations for the APS algorithm
        burnin : float, optional (default=0.75)
            Percentage of burn-in iterations in APS inner algorithms
    """
    a_sim = np.zeros(N_inner, dtype = int)
    a_sim[0] = np.random.choice(a_values)
    theta_sim = theta(d_given, a_sim[0])

    for i in range(1,N_inner):
        ## Update a
        a_tilde = propose(a_sim[i-1], a_values)
        theta_tilde = theta(d_given, a_tilde)

        num = a_util(a_tilde, theta_tilde)
        den = a_util(a_sim[i-1], theta_sim)

        if np.random.uniform() <= num/den:
            a_sim[i] = a_tilde
            theta_sim = theta_tilde
        else:
            a_sim[i] = a_sim[i-1]

    a_dist = a_sim[int(burnin*N_inner):]
    return(mode(a_dist)[0][0], a_dist)


def aps_atk_def(d_values, a_values, d_util, a_util, theta, N_aps=1000,
                burnin=0.75, N_inner = 4000, verbose=False):

    """ Computes the solution of an attacker-defender game using APS

        Parameters
        ----------
        d_values : array-like
            Strategies for the defender
        a_values : array-like
            Strategies for the attacker
        d_util : callable
            Computes the utility of the defender

                ``d_util(d, theta) -> array, shape (same as theta)``

            where theta is a 1-D array and d is the decisions of the defender
        a_util : callable
            Computes the utility of the attacker

                ``a_util(a, theta) -> array, shape (same as theta)``

            where theta is a 1-D array and a is the decision of the attacker
        theta : callable
            Draws samples of theta from the distribution p(theta | d, a)

                ``theta(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions for the defender and the attacker
        N_aps : int, optional (default=1000)
            Number of iterations for the APS algorithm
        burnin : float, optional (default=0.75)
            Percentage of burn-in iterations in the APS and APS inner algorithms
    """

    ## Compute a* for all d
    if verbose:
        print("PREPROCESSING...")
    a_star = np.zeros_like(d_values)
    a_dist = {}
    for i, d in enumerate(d_values):
        a_star[i], a_dist[d] = innerAPS(d, a_values, a_util, theta, N_inner, burnin)


    ## Start d_sim, vector of length N_aps to save in it simulation results
    d_sim = np.zeros(N_aps, dtype = int)
    d_sim[0] = np.random.choice(d_values)
    a_sim = a_star[d_values == d_sim[0]][0]
    theta_sim = theta(d_sim[0], a_sim)

    if verbose:
        print("START")

    for i in range(1,N_aps):
        ## Update d
        d_tilde = propose(d_sim[i-1], d_values)
        a_tilde = a_star[d_values == d_tilde][0]
        theta_tilde = theta(d_tilde, a_tilde)

        num = d_util(d_tilde, theta_tilde)

        den = d_util(d_sim[i-1], theta_sim)


        if np.random.uniform() <= num/den:
            d_sim[i] = d_tilde
            a_sim = a_tilde
            theta_sim = theta_tilde
        else:
            d_sim[i] = d_sim[i-1]

        if verbose:
            if i%1000 == 0:
                print(i)

    d_dist = d_sim[int(burnin*N_aps):]
    return(mode(d_dist)[0], a_star, a_dist, d_dist)


def aps_ara(d_values, a_values, d_util, a_util_f, theta_d, a_prob_f,
            N_aps=1000, burnin=0.75, N_inner = 1000, J = 100, verbose=False):
    """ Computes the solution of ARA using MCMC

        Parameters
        ----------
        d_values : array-like
            Strategies for the defender
        a_values : array-like
            Strategies for the attacker
        d_util : callable
            Computes the utility of the defender

                ``d_util(d, theta) -> array, shape (same as theta)``

            where theta is a 1-D array and d is the decision of the defender
        a_util_f : callable
            Returns a random function

                ``a_util_f() -> callable``

            with signature

                ``a_util(a, theta) -> array, shape (same as theta)``

            where theta is a 1-D array and a is the decision of the attacker
        theta_d : callable
            Draws samples of theta from the distribution p_d(theta | d, a)

                ``theta_d(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions for the defender and the attacker
        a_prob_f : callable
            Returns random function

                ``a_prob_f() -> callable``

            with signature

                ``a_prob(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions of the defender and the attacker
        N_aps : int, optional (default=1000)
            Number of iterations for the APS algorithm
        burnin : float, optional (default=0.75)
            Percentage of burn-in iterations in the APS and APS inner algorithms
        J : int, optional (default=100)
            Number of inner iterations to compute the random optimal
            attacks distribution
    """
    ## Compute a* for all d
    if verbose:
        print("PREPROCESSING...")
    p_d = np.zeros((len(d_values), len(a_values)), dtype=float)

    for i, d_given in enumerate(d_values):
        modes = np.zeros(J, dtype=int)
        for jj in range(J):
            modes[jj], _ = innerAPS(d_given, a_values, a_util_f(),
                                 a_prob_f(), N_inner, burnin )
        p_d[i, :] = np.bincount(modes, minlength = len(a_values))/J

    d_sim = np.zeros(N_aps, dtype = int)
    d_sim[0] = np.random.choice(d_values)
    a_sim = np.random.choice(a_values, p=p_d[d_values == d_sim[0], :][0])
    theta_sim = theta_d(d_sim[0], a_sim)

    if verbose:
        print("START")

    for i in range(1,N_aps):
        ## Update d
        d_tilde = propose(d_sim[i-1], d_values)
        a_tilde = np.random.choice(a_values, p=p_d[d_values == d_tilde, :][0])
        theta_tilde = theta_d(d_tilde, a_tilde)

        num = d_util(d_tilde, theta_tilde)

        den = d_util(d_sim[i-1], theta_sim)


        if np.random.uniform() <= num/den:
            d_sim[i] = d_tilde
            a_sim = a_tilde
            theta_sim = theta_tilde
        else:
            d_sim[i] = d_sim[i-1]

        if verbose:
            if i%1000 == 0:
                print(i)

    d_dist = d_sim[int(burnin*N_aps):]
    return(mode(d_dist)[0], p_d, d_dist)
