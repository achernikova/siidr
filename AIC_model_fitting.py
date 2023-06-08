import numpy as np
import pandas as pd
import igraph as ig
from igraph import *
import random
import sys
import os
import matplotlib.pyplot as plt
import scipy
from collections import defaultdict
from scipy.sparse import linalg
from  scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs, svds
import collections
import argparse
import math
import time
from scipy import stats
from collections import OrderedDict
from sklearn.model_selection import ParameterGrid
import multiprocessing
from joblib import Parallel, delayed

import utils
from SPM_stochastic_simulation import Simulation

#filenames of WannaCry variants
variants = [el for el in os.listdir('wc_data/') if (el != '.DS_Store' and el != '.ipynb_checkpoints' and el != 'wc_128_100ms_10vm.csv' and el != 'wc_128_100ms_10vm.csv' and el != 'wc_128_100ms_10vm.log' and el != 'wc_128_100ms_50vm_host1.log' and el != 'conn_50vms_10s_network.csv')]

# dictionary of results
results = dict()

# iterate over variants
for variant in variants:

    # results for this variant
    results[variant] = dict()

    # import data
    df = utils.import_data(variant)

    # loc all malicious traffic towards internal network
    df_p = df.loc[(df['id.resp_p'] == 445) & (df['id.resp_h'].str.startswith('192.168')) & (df['id.orig_h'].str.startswith('192.168'))]

    delta_activity   = [] # delta t between two attacks from the same IP
    inactivity_times = [] # delta t between last attack and end of simulation
    malicious_IP     = df_p['id.orig_h'].unique() # all malicious ip

    for ip in malicious_IP:

        df_ip = df_p.loc[df_p['id.orig_h'] == ip]   # loc traffic for this specific ip
        de    = df_ip.ts.diff().dropna()          # delta t in activity times

        for d in de:
            delta_activity.append(d)

        inactivity_times.append(df.ts.values[-1] - df_ip.ts.values[-1])

    results[variant]['delta_activity']   = delta_activity
    results[variant]['inactivity_times'] = inactivity_times

    # reconstruct epidemics
    ts, I, spont_I, infected_IP, contacted_IP = utils.epi_wcry(df, True)

    results[variant]['ts']           = ts
    results[variant]['I']            = I
    results[variant]['spont_I']      = spont_I
    results[variant]['infected_IP']  = infected_IP
    results[variant]['contacted_IP'] = contacted_IP

for variant in variants:

    print(variant)
    print('\tspontaneous infections:', results[variant]['spont_I'])
    print('\tN. contacted IP:', len(results[variant]['contacted_IP']))
    print('\tN. infected IP:', results[variant]['I'][-1])
    print('\tPerc. infected IP: %.2f' % (results[variant]['I'][-1] / len(results[variant]['contacted_IP'])))


variants_plotlist = ['wc_1_500ms.csv',
                     'wc_1_1s.csv'   ,
                     'wc_1_5s.csv'   ,
                     'wc_1_10s.csv'  ,
                     'wc_1_20s.csv'  ,
                     'wc_4_500ms.csv',
                     'wc_4_1s.csv'   ,
                     'wc_4_5s.csv'   ,
                     'wc_4_10s.csv'  ,
                     'wc_4_20s.csv'  ,
                     'wc_8_500ms.csv',
                     'wc_8_1s.csv'   ,
                     'wc_8_5s.csv'   ,
                     'wc_8_10s.csv'  ,
                     'wc_8_20s.csv']

#Model fitting with AIC

## SI AIC model
# number of free parameters
k = 1

# n. of stochastic simulations
nsim = 20

# dict of results for model1
model1 = dict()

# create lists of parameters
beta   = np.linspace(start = 0.01, stop = 0.99, num = 20)
mu = [0]
gamma1 = [0]
gamma2 = [0]

# n. of equidistant points considered from the actual dynamics
n = [100]

R0     = [0]
I0     = [1]

# iterate over variants
for variant in variants:

    print('Variant:', variant)

    # get the n evenly spaced point from the actual I dynamics
    # (this speeds up significantly the simulations)
    actual_I = np.array(results[variant]['I'])
    idx      = np.round(np.linspace(0, len(actual_I) - 1, n[0])).astype(int)
    actual_I = actual_I[idx]

    # set variant specific variables
    N  = [len(results[variant]['contacted_IP'])]
    dt = [(results[variant]['ts'][-1]-results[variant]['ts'][0]) / n[0]]


    # create grid of parameters
    param_grid = ParameterGrid({'beta'  : beta,
                                'mu'    : mu,
                                'gamma1': gamma1,
                                'gamma2': gamma2,
                                'N'     : N,
                                'dt'    : dt,
                                'n'     : n,
                                'R0'    : R0,
                                'I0'    : I0})

    # simulate
    sim = Simulation(param_grid, actual_I, k)
    sim.parallel_simulation(nsim, 'siidr')
    sim.get_best_models()

    model1[variant] = sim

    print('best aic siidr', min(sim.aic))
    print('best params beta', sim.best_params['beta'])
    print('best params mu', sim.best_params['mu'])
    print('best params gamma1', sim.best_params['gamma1'])
    print('best params gamma2', sim.best_params['gamma2'])

### SIR
# number of free parameters
k = 2

# n. of stochastic simulations
nsim = 20

# dict of results for model1
model2 = dict()

# create lists of parameters
beta   = np.linspace(start = 0.01, stop = 0.99, num = 20)
mu     = np.linspace(start = 0.01, stop = 0.99, num = 20)
gamma1 = [0]
gamma2 = [0]

# n. of equidistant points considered from the actual dynamics
n = [100]

R0     = [0]
I0     = [1]

# iterate over variants
for variant in variants:

    print('Variant:', variant)

    # get the n evenly spaced point from the actual I dynamics
    # (this speeds up significantly the simulations)
    actual_I = np.array(results[variant]['I'])
    idx      = np.round(np.linspace(0, len(actual_I) - 1, n[0])).astype(int)
    actual_I = actual_I[idx]

    # set variant specific variables
    N  = [len(results[variant]['contacted_IP'])]
    dt = [(results[variant]['ts'][-1]-results[variant]['ts'][0]) / n[0]]

    # create grid of parameters
    param_grid = ParameterGrid({'beta'  : beta,
                                'mu'    : mu,
                                'gamma1': gamma1,
                                'gamma2': gamma2,
                                'N'     : N,
                                'dt'    : dt,
                                'n'     : n,
                                'R0'    : R0,
                                'I0'    : I0})

    # simulate
    sim = Simulation(param_grid, actual_I, k)

    sim.parallel_simulation(nsim, 'sir')

    sim.get_best_models()

    model2[variant] = sim

    print('best aic siidr', min(sim.aic))
    print('best params beta', sim.best_params['beta'])
    print('best params mu', sim.best_params['mu'])
    print('best params gamma1', sim.best_params['gamma1'])
    print('best params gamma2', sim.best_params['gamma2'])

## SIS
# number of free parameters
k = 2

# n. of stochastic simulations
nsim = 20

# dict of results for model1
model3 = dict()

# create lists of parameters
beta   = np.linspace(start = 0.01, stop = 0.99, num = 20)
mu     = np.linspace(start = 0.01, stop = 0.99, num = 20)
gamma1 = [0]
gamma2 = [0]

# n. of equidistant points considered from the actual dynamics
n = [100]

R0     = [0]
I0     = [1]

# iterate over variants
for variant in variants:

    print('Variant:', variant)

    # get the n evenly spaced point from the actual I dynamics
    # (this speeds up significantly the simulations)
    actual_I = np.array(results[variant]['I'])
    idx      = np.round(np.linspace(0, len(actual_I) - 1, n[0])).astype(int)
    actual_I = actual_I[idx]

    # set variant specific variables
    N  = [len(results[variant]['contacted_IP'])]
    dt = [(results[variant]['ts'][-1]-results[variant]['ts'][0]) / n[0]]

    # create grid of parameters
    param_grid = ParameterGrid({'beta'  : beta,
                                'mu'    : mu,
                                'gamma1': gamma1,
                                'gamma2': gamma2,
                                'N'     : N,
                                'dt'    : dt,
                                'n'     : n,
                                'R0'    : R0,
                                'I0'    : I0})

    sim = Simulation(param_grid, actual_I, k)

    sim.parallel_simulation(nsim, 'sis')

    sim.get_best_models()

    model3[variant] = sim

    print('best aic siidr', min(sim.aic))
    print('best params beta', sim.best_params['beta'])
    print('best params mu', sim.best_params['mu'])
    print('best params gamma1', sim.best_params['gamma1'])
    print('best params gamma2', sim.best_params['gamma2'])

## SIIDR
# number of free parameters
k = 4

# n. of stochastic simulations
nsim = 20

# dict of results for model1
model4 = dict()

# create lists of parameters
beta   = np.linspace(start = 0.01, stop = 0.99, num = 20)
mu     = np.linspace(start = 0.01, stop = 0.99, num = 20)
gamma1 = np.linspace(start = 0.01, stop = 0.99, num = 10)
gamma2 = np.linspace(start = 0.01, stop = 0.99, num = 10)

# n. of equidistant points considered from the actual dynamics
n = [100]

R0     = [0]
I0     = [1]

# iterate over variants
for variant in variants:

    print('Variant:', variant)

    # get the n evenly spaced point from the actual I dynamics
    # (this speeds up significantly the simulations)
    actual_I = np.array(results[variant]['I'])
    idx      = np.round(np.linspace(0, len(actual_I) - 1, n[0])).astype(int)
    actual_I = actual_I[idx]

    # set variant specific variables
    N  = [len(results[variant]['contacted_IP'])]
    dt = [(results[variant]['ts'][-1]-results[variant]['ts'][0]) / n[0]]

    # create grid of parameters
    param_grid = ParameterGrid({'beta'  : beta,
                                'mu'    : mu,
                                'gamma1': gamma1,
                                'gamma2': gamma2,
                                'N'     : N,
                                'dt'    : dt,
                                'n'     : n,
                                'R0'    : R0,
                                'I0'    : I0})

    sim = Simulation(param_grid, actual_I, k)

    sim.parallel_simulation(nsim, 'siidr')

    sim.get_best_models()

    model4[variant] = sim

    print('best aic siidr', min(sim.aic))
    print('best params beta', sim.best_params['beta'])
    print('best params mu', sim.best_params['mu'])
    print('best params gamma1', sim.best_params['gamma1'])
    print('best params gamma2', sim.best_params['gamma2'])



#plot model fitting results
plt.figure(figsize = (16,9))
for i in range(len(variants_plotlist)):

    sim4 = model4[variants_plotlist[i]]
    aic_min4 = np.min(sim4.aic)
    idx_min4 = np.argmin(sim4.aic)

    sim3 = model3[variants_plotlist[i]]
    aic_min3 = np.min(sim3.aic)
    idx_min3 = np.argmin(sim3.aic)

    sim2 = model2[variants_plotlist[i]]
    aic_min2 = np.min(sim2.aic)
    idx_min2 = np.argmin(sim2.aic)

    sim1 = model1[variants_plotlist[i]]
    aic_min1 = np.min(sim1.aic)
    idx_min1 = np.argmin(sim1.aic)

    plt.subplot(3, 5, i+1)
    plt.plot(sim4.actual_I, label = "Actual", marker='D', markersize=4, markevery=6, color = 'lightseagreen',  linewidth=1.5)
    plt.plot(sim4.tot_avg[idx_min4], label = "SIIDR",  linewidth=1.5, color = 'orangered')
    plt.plot(sim3.tot_avg[idx_min3], label = "SIS",  linewidth=1.5, color = 'gold')
    plt.plot(sim2.tot_avg[idx_min2], label = "SIR",  linewidth=1.5, color = 'yellowgreen')
    plt.plot(sim1.tot_avg[idx_min1], label = "SI",  linewidth=1.5, color = 'palevioletred')
    plt.title(variants_plotlist[i][0:len(variants_plotlist[i])-4], fontsize=20)
    plt.xlabel('$t$', fontsize=15)
    plt.ylabel('$I_{t}$', fontsize=15)
    plt.legend()

plt.tight_layout(),
my_dpi = 300
plt.savefig('aic_simulation.png', dpi = my_dpi)