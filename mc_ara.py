#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from importlib import import_module

sys.path.append('.')

def solve_attacker_mc(d, a_util, a_prob, iters):
    psi_a = np.zeros((len(p.a_values)), dtype=float)
    for i, a in enumerate(p.a_values):
        theta = a_prob(a, d, size=iters)
        psi_a[i] = a_util(a, theta).mean()
    a_opt = p.a_values[psi_a.argmax()]
    return a_opt

def ara_iter(d, K, iters):

    def wrapper():
        ##
        params = p.sample_params()
        a_prob = lambda a, d, size: p.prob_a(a, d, params, size=size)
        a_util = lambda a, theta: p.ua(a, theta, params)
        ##
        return solve_attacker_mc(d, a_util, a_prob, iters)

    with Parallel(n_jobs=-1) as parallel:
        result = parallel(delayed(wrapper)() for k in range(K))

    result = np.array(result)
    return np.bincount(result.astype('int'), minlength=len(p.a_values))  / K

def compute_p_ad(K, iters):
    p_ad = np.zeros([len(p.d_values),len(p.a_values)])
    ##
    for i,d in enumerate(p.d_values):
        p_ad[i] = ara_iter(d, K, iters)
    return p_ad

if __name__ == '__main__':

    K = 10
    iters = 10
    p = import_module(f'data.prob2_new')
    p_ad = compute_p_ad(K, iters)
    df = pd.DataFrame(p_ad)
    df.columns = p.a_values
    df.index = p.d_values
    df.to_csv('results/p_ad.csv')
