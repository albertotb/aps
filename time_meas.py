import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import prob_new as p
from aps_annealing import *
from time_mc import *
from mcmc import *
from joblib import Parallel, delayed
import multiprocessing
import sys
import pickle
import argparse
from timeit import default_timer
from importlib import import_module
from contextlib import contextmanager

times = []
for i in range(1000):
    start = default_timer()
    ##
    d_opt = aps_adg_ann(100000, 1000, d_util, a_util, prob, N_aps=1000, burnin=0.1, N_inner=100, prec=0.01, mean=True, 
                     info=False)
    ##
    end = default_timer()
    times.append(end - start)

print(np.mean(times), np.std(times))
