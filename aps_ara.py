import numpy as np
import pandas as pd
import sys
from importlib import import_module

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

def iter_mcmc(J, d_sim, theta_sim):
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

        d_sim[i], theta_sim = iter_mcmc(J, d_sim[i-1], theta_sim)
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

        d_sim[i], theta_sim = iter_mcmc(J, d_sim[i-1], theta_sim)
        theta_sim = np.append(theta_sim, np.random.choice(theta_sim))
        if J%500 == 0:
            print(i)
            burnin = int(0.2*i)
            dist = pd.Series(d_sim[burnin:i])
            name = 'results/dist_APS_J' + str(J) + '.csv'
            dist.to_csv(name, index = False)

if __name__ == '__main__':

    p = import_module(f'data.prob2_new')
    p_ad = pd.read_csv('results/p_ad.csv', index_col='Unnamed: 0').values

    N = 10000000
    solve_defender_nt(N, p_ad, propose)

    #J_max = 10000
    # solve_defender(J_max, p_ad, propose)
