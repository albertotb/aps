#!/usr/bin/env python
import sys
import pickle
import numpy as np
import pandas as pd
import time
import typer
from itertools import product
from timeit import default_timer
from importlib import import_module
from contextlib import contextmanager

sys.path.append('.')
from mcmc import mcmc_adg, mcmc_ara
from aps import aps_adg, aps_ara
from aps_annealing import aps_adg_ann


@contextmanager
def timer():
    start = default_timer()
    try:
        yield
    finally:
        end = default_timer()
        print("Elapsed time (s): {:.6f}".format(end - start))


app = typer.Typer()
mcmc_app = typer.Typer()
aps_app = typer.Typer()
ann_app = typer.Typer()
app.add_typer(mcmc_app, name="mcmc")
app.add_typer(aps_app,  name="aps")
app.add_typer(ann_app,  name="aps_ann")

state = {}


@app.callback()
def global_args(prob: str = "prob1", out: str = "./results"):

    ts_sec = int(time.time())
    state['p'] = import_module(f'data.{prob}')
    state['d_idx'] = pd.Index(state['p'].d_values, name='d')
    state['a_idx'] = pd.Index(state['p'].a_values, name='a')
    state['cols'] = map(lambda x: f'{x[0]}_{x[1]}',
                        product(['mean', 'std'], state['a_idx']))
    state['basepath'] = f'{out}/{ts_sec}_{prob}'


@mcmc_app.command("adg")
def cli_mcmc_adg(iters: int = 100, inner_iters: int = 100):

    p = state['p']

    with timer():
        (d_opt, a_opt,
         psi_d, psi_d_std,
         psi_a, psi_a_std, t) = mcmc_adg(p.d_values, p.a_values, p.d_util,
                                         p.a_util, p.prob, p.prob,
                                         iters=iters, inner_iters=inner_iters)

    psi_d = pd.DataFrame({'mean': psi_d, 'std': psi_d_std},
                         index=state['d_idx'])
    psi_a = pd.DataFrame(np.concatenate((psi_a, psi_a_std), axis=1),
                         index=state['d_idx'], columns=state['cols'])

    psi_d.to_csv(f'{state["basepath"]}_mcmc_adg_psid.csv')
    psi_a.to_csv(f'{state["basepath"]}_mcmc_adg_psia.csv')

    typer.echo(a_opt)
    typer.echo(d_opt)


@aps_app.command("adg")
def cli_aps_adg(iters: int = 10, inner_iters: int = 10, burnin: float = 0.75):

    p = state['p']

    with timer():
        d_opt, a_opt, psi_d, psi_a = aps_adg(p.d_values, p.a_values, p.d_util,
                                             p.a_util, p.prob, burnin=burnin,
                                             N_aps=iters, N_inner=inner_iters)

    pd.Series(psi_d).to_csv(f'{state["basepath"]}_aps_adg_psid.csv',
                             header=False, index=False)

    psi_a = pd.DataFrame(psi_a)
    psi_a.columns = p.d_values
    pd.DataFrame(psi_a).to_csv(f'{state["basepath"]}_aps_adg_psia.csv',
                             header=True, index=False)


@mcmc_app.command("ara")
def cli_mcmc_ara(iters: int = 100, ara_iters: int = 10, n_jobs: int = -1):

    p = state['p']

    with timer():
        (d_opt, psi_d, p_ad) = mcmc_ara(p.d_values, p.a_values, p.d_util,
                                        p.a_util_f, p.prob, p.a_prob_f,
                                        iters=iters, ara_iters=ara_iters,
                                        n_jobs=n_jobs)

    psi_d = pd.DataFrame(psi_d, index=state["d_idx"])
    p_ad = pd.DataFrame(p_ad, index=state["d_idx"], columns=state["a_idx"])

    psi_d.to_csv(f'{state["basepath"]}_mcmc_ara_psid.csv')
    p_ad.to_csv(f'{state["basepath"]}_mcmc_ara_pa.csv')

    '''
    psi_d = pd.DataFrame({'mean': psi_da.sum(axis=1),
                          'std': psi_da_std.sum(axis=1)}, index=state["d_idx"])
    psi_a = pd.DataFrame(np.concatenate((
        psi_ad.mean(axis=2),
        psi_ad_std.mean(axis=2)),
        axis=1), index=state["d_idx"], columns=state["cols"])

    psi_a.to_csv(f'{state["basepath"]}_mcmc_ara_psia.csv')
    a_opt = pd.Series(p_a.argmax(axis=1), index=state["d_idx"])

    typer.echo(a_opt)
    typer.echo(d_opt)
    '''





@aps_app.command("ara")
def cli_aps_ara(iters: int = 10, inner_iters: int = 10, burnin: float = 0.20,
                ara_iters: int = 10, pa: str = None, n_jobs: int = -1):

    p = state['p']

    if pa is not None:
        p_ad = pd.read_csv(pa).values
    else:
        p_ad = None


    with timer():

        d_opt, p_ad, psi_d = aps_ara(p.d_values, p.a_values, p.d_util,
                                     p.a_util_f, p.prob, p.a_prob_f,
                                     N_aps=iters, burnin=burnin,
                                     N_inner=inner_iters, ara_iters=ara_iters,
                                     n_jobs = n_jobs, p_ad=p_ad)

    pd.Series(psi_d).to_csv(f'{state["basepath"]}_aps_ara_psid.csv',
                             header=False, index=False)

    p_ad = pd.DataFrame(p_ad, index=state["d_idx"], columns=state["a_idx"])
    p_ad.to_csv(f'{state["basepath"]}_aps_ara_pa.csv')


@ann_app.command("adg")
def cli_ann_adg(iters: int = 10, inner_iters: int = 10, burnin: float = 0.75,
                temp: int = 5, prec: float = 0.01, mean: bool = True):

    p = state['p']

    with timer():
        d_opt, psi_d = aps_adg_ann(p.d_util, p.a_util, p.prob, J=temp,
                                   J_inner=temp, N_aps=iters, burnin=burnin,
                                   N_inner=inner_iters, prec=prec, mean=mean,
                                   info=True)


if __name__ == '__main__':
    app()
