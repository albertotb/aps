import numpy as np
import pandas as pd
import sys
from joblib import Parallel, delayed
from scipy.stats import mode
from importlib import import_module
import os.path

sys.path.append('.')

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

def propose2(a_sim, a_values):
    return np.random.choice(a_values)

def iter_mcmc_att(J, propose, a_sim, theta_sim, d, a_util, a_prob):
    a_tilde = propose(a_sim, p.a_values)
    theta_tilde = a_prob(a_tilde, d, size=J)

    num = a_util(a_tilde, theta_tilde)
    den = a_util(a_sim, theta_sim)
    quot = np.prod(num/den)


    if np.random.uniform() <= quot:
        return a_tilde, theta_tilde
    else:
        return a_sim, theta_sim

def iter_mcmc_defender(J, propose, d_sim, theta_sim, p_ad):
    d_tilde = propose(d_sim, p.d_values)
    a_tilde = np.random.choice(p.a_values, p=p_ad[p.d_values == d_tilde, :][0], size = J)
    theta_tilde = np.array([p.prob_d(a, d_tilde) for a in a_tilde])[:,0]

    num = p.ud(d_tilde, theta_tilde)

    den = p.ud(d_sim, theta_sim)

    quot = np.prod(num/den)

    if np.random.uniform() <= quot:
        return d_tilde, theta_tilde
    else:
        return d_sim, theta_sim

def solve_attacker_aps(d, a_util, a_prob, J_max, propose):
    J_grid = range(1, J_max)
    a_sim = np.zeros(len(J_grid), dtype = int)
    a_sim[0] = np.random.choice(p.a_values)
    theta_sim = a_prob(a_sim[0], d, size=1)
    ##
    for i, J in enumerate(J_grid):
        a_sim[i], theta_sim = iter_mcmc_att(J, propose, a_sim[i-1], theta_sim, d, a_util, a_prob)
        idx = np.random.choice( np.arange(0, theta_sim.shape[0]) )
        theta_sim = np.vstack( [theta_sim, theta_sim[idx]] )
        #if i%500 == 0:
        #    burnin = int(0.2*i)
        #    print( mode( a_sim[burnin:i] )[0] )

    burnin = int(0.2*J_max)
    m =  mode(a_sim[burnin:])[0]
    return int(m)

def ara_iter(d, K, iters, propose):

    def wrapper():
        ##
        params = p.sample_params()
        a_prob = lambda a, d, size: p.prob_a(a, d, params, size=size)
        a_util = lambda a, theta: p.ua(a, theta, params)
        ##
        return solve_attacker_aps(d, a_util, a_prob, iters, propose)

    with Parallel(n_jobs=-1) as parallel:
        result = parallel(delayed(wrapper)() for k in range(K))

    result = np.array(result)
    return np.bincount(result.astype('int'), minlength=len(p.a_values))  / K

def compute_p_ad(K, iters, propose):
    p_ad = np.zeros([len(p.d_values),len(p.a_values)])
    ##
    for i,d in enumerate(p.d_values):
        p_ad[i] = ara_iter(d, K, iters, propose)
    return p_ad

# Without Muller's trick
def solve_defender_nt(N, p_ad, propose):
    J = 1
    N_grid = range(1, N)
    d_sim = np.zeros(len(N_grid), dtype = int)
    d_sim[0] = np.random.choice(p.d_values)
    a_sim = np.random.choice(p.a_values, p=p_ad[p.d_values == d_sim[0], :][0], size = J)
    theta_sim = np.array([p.prob_d(a, d_sim[0]) for a in a_sim])[:,0]
    ##
    for i, n in enumerate(N_grid):

        d_sim[i], theta_sim = iter_mcmc(J, propose, d_sim[i-1], theta_sim, p_ad)
        if n%10000 == 0:
            print( n/N )

    # burnin = int(0.5*N)
    dist = pd.Series(d_sim)
    name = 'results/dist_APS_no_Jtrick.csv'
    dist.to_csv(name, index = False)


# With Muller's trick
def solve_defender(J_max, p_ad, propose):
    J = 1
    J_grid = range(1, J_max)
    d_sim = np.zeros(len(J_grid), dtype = int)
    d_sim[0] = np.random.choice(p.d_values)
    a_sim = np.random.choice(p.a_values, p=p_ad[p.d_values == d_sim[0], :][0], size = J)
    theta_sim = np.array([p.prob_d(a, d_sim[0]) for a in a_sim])[:,0]
    ##
    for i, J in enumerate(J_grid):

        d_sim[i], theta_sim = iter_mcmc(J, propose, d_sim[i-1], theta_sim, p_ad)
        theta_sim = np.append(theta_sim, np.random.choice(theta_sim))
        if J%500 == 0:
            print(i)
            burnin = int(0.2*i)
            dist = pd.Series(d_sim[burnin:i])
            name = 'results/dist_APS_J' + str(J) + '.csv'
            dist.to_csv(name, index = False)

if __name__ == '__main__':

    p = import_module(f'data.prob2_new')

    if not os.path.isfile('results/p_ad.csv'):

        K=1000
        iters=2000
        p_ad = compute_p_ad(K, iters, propose2)
        df = pd.DataFrame(p_ad)
        df.columns = p.a_values
        df.index = p.d_values
        df.to_csv('results/p_ad.csv')

    #
    p_ad = pd.read_csv('results/p_ad.csv', index_col='Unnamed: 0').values
    
    # N = 10000000
    # solve_defender_nt(N, p_ad, propose)

    J_max = 10000
    solve_defender(J_max, p_ad, propose)
