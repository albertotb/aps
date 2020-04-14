#!/usr/bin/env python
from timeit import default_timer
import numpy as np
from math import sqrt
from joblib import Parallel, delayed


def mcmc_adg(d_values, a_values, d_util, a_util, d_prob, a_prob,
             iters=1000, inner_iters=1000):
    """ Computes the solution of an attacker-defender game using MCMC

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
        d_prob : callable
            Draws samples of theta from the distribution p_d(theta | d, a)

                ``d_prob(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions for the defender and the attacker
        a_prob : callable
            Draws samples of theta from the distribution p_a(theta | d, a)

                ``a_prob(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions for the defender and the attacker
        n : int, optional (default=1000)
            Number of iterations for the Montecarlo algorithm
    """
    a_opt = np.zeros_like(d_values)
    psi_d = np.zeros_like(d_values, dtype=float)
    psi_d_std = np.zeros_like(d_values, dtype=float)
    psi_a = np.zeros((len(d_values), len(a_values)), dtype=float)
    psi_a_std = np.zeros((len(d_values), len(a_values)), dtype=float)
    times = np.zeros_like(d_values, dtype=float)

    for i, d in enumerate(d_values):
        start = default_timer()
        for j, a in enumerate(a_values):
            theta_a = a_prob(d, a, size=inner_iters)
            psi_a[i, j] = a_util(a, theta_a).mean()
            psi_a_std[i, j] = a_util(a, theta_a).std() / sqrt(inner_iters)
        a_opt[i] = a_values[psi_a[i, :].argmax()]
        end = default_timer()
        theta_d = d_prob(d, a_opt[i], size=iters)
        psi_d[i] = d_util(d, theta_d).mean()
        psi_d_std[i] = d_util(d, theta_d).std() / sqrt(iters)
        times[i] = end - start

    d_opt = d_values[psi_d.argmax()]
    return d_opt, a_opt, psi_d, psi_d_std, psi_a, psi_a_std, times


def mcmc_ara(d_values, a_values, d_util, a_util_f, d_prob, a_prob_f,
             iters=1000, ara_iters=1000, n_jobs=1, p_ad=None):
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
        d_prob : callable
            Draws samples of theta from the distribution p_d(theta | d, a)

                ``d_prob(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions for the defender and the attacker
        a_prob_f : callable
            Returns random function

                ``a_prob_f() -> callable``

            with signature

                ``a_prob(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions of the defender and the attacker
        n : int, optional (default=1000)
            Number of outer iterations for the Montecarlo algorithm
        m : int, optional (default=100)
            Number of inner iterations for the Montecarlo algorithm
    """
    def solve_attacker_mc(d, a_util, a_prob, iters):
        #
        psi_a = np.zeros((len(a_values)), dtype=float)
        for i, a in enumerate(a_values):
            theta = a_prob(d, a, size=iters)
            psi_a[i] = a_util(a, theta).mean()
        a_opt = a_values[psi_a.argmax()]
        return a_opt

    def ara_iter(d, K, iters):

        def wrapper():
            ##
            a_prob = a_prob_f()
            a_util = a_util_f()
            ##
            return solve_attacker_mc(d, a_util, a_prob, iters)

        with Parallel(n_jobs=n_jobs) as parallel:
            result = parallel(delayed(wrapper)() for k in range(K))

        result = np.array(result)
        return np.bincount(result.astype('int'), minlength=len(a_values))  / K

    def compute_p_ad(K, iters):
        p_ad = np.zeros([len(d_values),len(a_values)])
        ##
        for i,d in enumerate(d_values):
            p_ad[i] = ara_iter(d, K, iters)
        return p_ad

    if p_ad is None:
        p_ad = compute_p_ad(ara_iters, iters)
    else:

    psi_d = np.zeros((len(d_values)), dtype=float)
    ##
    for i, d in enumerate(d_values):
        sample = np.random.choice(a_values, p = p_ad[i], size = iters)
        theta_d = np.array([d_prob(d, a) for a in sample])[:,0]
        psi_d[i] = d_util(d, theta_d).mean()
    d_opt = d_values[psi_d.argmax()]


    return d_opt, psi_d, p_ad
