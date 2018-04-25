#!/usr/bin/env python
import numpy as np
from scipy.integrate import quad

def scale(x):
    return (x - x.min())/(x.max() - x.min())

def mcmc_atk_def(d_values, a_values, d_util, a_util, prob, n=1000):
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
        prob : callable
            Draws samples of theta from the distribution p(theta | d, a)

                ``prob(d, a, size=1) -> array, shape (size,)``

            where d and a are the decisions for the defender and the attacker
        n : int, optional (default=1000)
            Number of iterations for the Montecarlo algorithm
    """
    a_opt = np.zeros_like(d_values)
    psi_d = np.zeros_like(d_values, dtype=float)
    for i, d in enumerate(d_values):
        psi_a = np.zeros_like(a_values, dtype=float)
        for j, a in enumerate(a_values):
            theta_a = prob(d, a, size=n)
            psi_a[j] = a_util(a, theta_a).mean()
        a_opt[i] = a_values[psi_a.argmax()]
        theta_d = prob(d, a_opt[i], size=n)
        psi_d[i] = d_util(d, theta_d).mean()

    d_opt = d_values[psi_d.argmax()]
    return d_opt, a_opt, psi_a, psi_d


def mcmc_ara(d_values, a_values, d_util, a_util_f, d_prob, a_prob_f, n=1000, m=1000):
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
    psi_d = np.zeros((len(d_values), len(a_values)), dtype=float)
    p_d = np.zeros((len(d_values), len(a_values)), dtype=float)

    for i, d in enumerate(d_values):
        psi_a = np.zeros((len(a_values), m), dtype=float)
        for j, a in enumerate(a_values):
            for k in range(m):
                a_util = a_util_f()
                a_prob = a_prob_f(d)
                theta_k = a_prob(d, a, size=n)
                psi_a[j, k] = a_util(a, theta_k).mean()

        p_d[i, :] = np.bincount(psi_a.argmax(axis=0), minlength=len(a_values))/m

        for j, a in enumerate(a_values):
            theta_d = d_prob(d, a, size=n)
            psi_d[i, j] = (d_util(d, theta_d)*p_d[i, j]).mean()

    d_opt = d_values[psi_d.sum(axis=1).argmax()]
    return d_opt, p_d, psi_a, psi_d
