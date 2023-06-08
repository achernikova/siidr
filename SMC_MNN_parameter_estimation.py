from scipy.stats import truncnorm
from numpy.linalg import inv
import numpy as np
from scipy.stats import uniform
from scipy.stats import multivariate_normal
import os
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

import utils

#stochastic simulations of SIIDR
def stochastic(beta, mu, gamma1, gamma2, N, dt, n, R0, I0):

    S  = [N - I0 - R0]
    I  = [I0]
    ID = [0]
    R  = [R0]
    t  = [0]

    for i in range(n - 1):
        if I[-1] == 0 and ID[-1] == 0:
            for j in np.arange(i, n - 1, 1):
                S.append(S[-1])
                I.append(I[-1])
                ID.append(ID[-1])
                R.append(R[-1])
                t.append(t[-1] + dt)
            break

        new_inf  = np.random.binomial(S[-1], 1.0 - np.exp(-beta * I[-1] / N * dt))

        I_leaving = np.random.binomial(I[-1], 1.0 - np.exp(- (mu + gamma1) * dt))

        if I_leaving != 0:
            p_R  = mu     / (mu + gamma1)
            p_ID = gamma1 / (mu + gamma1)

            I_multi  = np.random.multinomial(I_leaving, [p_R, p_ID])
            new_rec  = I_multi[0]
            new_dorm = I_multi[1]

        else:

            new_rec  = 0
            new_dorm = 0

        new_act  = np.random.binomial(ID[-1], 1.0 - np.exp(-gamma2 * dt))

        S.append(S[-1]   - new_inf)
        I.append(I[-1]   + new_inf  + new_act - new_rec - new_dorm)
        ID.append(ID[-1] + new_dorm - new_act)
        R.append(R[-1]   + new_rec)

    R = np.array(R) - R0

    return np.array(S), np.array(I), np.array(ID), np.array(R)

def simulation(param, nsim, actual, plot):

    beta   = param['beta']
    mu     = param['mu']
    gamma1 = param['gamma1']
    gamma2 = param['gamma2']
    N      = param['N_contacted']
    dt     = param['dt']
    n      = param['n']
    R0     = param['R0']
    I0     = param['I0']

    tot_I = 0

    S, I, ID, R = [], [], [], []

    for i in range(nsim):

        S_i, I_i, ID_i, R_i = stochastic(beta, mu, gamma1, gamma2, N, dt, n, R0, I0)
        S.append(S_i)
        I.append(I_i)
        ID.append(ID_i)
        R.append(R_i)


    tot_I   = np.array(I) + np.array(ID) + np.array(R)
    tot_avg = tot_I.mean(axis = 0)

    if plot:
        print("stochastic simulations")
        for i in range(nsim):
            plt.plot(tot_I[i])
        plt.plot(actual, label = 'actual dynamics')
        plt.legend()
        plt.show()

    return tot_avg

#distance between simulated and actual dynamics
def calculate_distance_between_models(infected_sim, infected_actual):

    res = 0

    for i in range(len(infected_sim)):
        res += (infected_sim[i] - infected_actual[i]) * (infected_sim[i] - infected_actual[i])

    return res

#checking whether the prior is valid
def prior_non_zero(params, lower_bound, upper_bound):

    result = 1

    for i in range(len(params)):
        result *= H(params[i]-lower_bound[i]) * H(upper_bound[i]-params[i])

    return result

#identity function
def H(x):
    if x > 0:
        return 1
    else:
        return 0

#sampling from truncated multivariate normal distribution
def sample_truncated_multivariate_normal_for_kernel(mean, sigma, lower_bound, upper_bound, n_samples = 1):

    samples = np.zeros((0, len(lower_bound)))

    while samples.shape[0] < n_samples:

        s = np.random.multivariate_normal(mean, sigma, size = (n_samples,))

        accepted = s[(np.min(s - lower_bound, axis = 1) >= 0) & (np.max(s - upper_bound, axis = 1) <= 0)]
        samples = np.concatenate((samples, accepted), axis = 0)

    samples = samples[:n_samples, :]

    return samples

#truncated multivariate normal pdf
def prob_trunc_norm(x, mean, sigma, lower_bound, upper_bound):

    if np.any(x < lower_bound) or np.any(x > upper_bound):
        return 0
    else:
        return multivariate_normal.pdf(x, mean, sigma)

#calculating sigma for the kernel based on M nearest neighbors
def get_sigma_neighbors(M, sampled_prev, all_prev):

    #calculate the euclidian distance between sampled examples between selected particle from previous generation
    #and all particles from previous generation
    distances = []

    for i in range(len(all_prev)):
        distance = np.sqrt(np.sum((np.array(all_prev[i]) - np.array(sampled_prev)) * (np.array(all_prev[i]) - np.array(sampled_prev))))
        distances.append(distance)

    #select M closest neighbors based on distance
    indices = np.array(distances).argsort()[1:(M+1)]
    indices_all = np.array(distances).argsort()[0:(M+1)]

    prev_closest = []

    for j in indices:
        prev_closest.append(all_prev[j])

    #calculate the covariance from M nearest neighbors
    cov = np.cov(np.transpose(np.array(prev_closest)))

    return cov

#variants = [el for el in os.listdir('wc_data/') if (el != '.DS_Store' and el != '.ipynb_checkpoints' and el != 'wc_128_100ms_10vm.csv' and el != 'wc_128_100ms_10vm.csv' and el != 'wc_128_100ms_10vm.log' and el != 'wc_128_100ms_50vm_host1.log' and el != 'conn_50vms_10s_network.csv')]
variants = ['wc_8_500ms.csv']
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

    # reconstruct epi
    ts, I, spont_I, infected_IP, contacted_IP = utils.epi_wcry(df, True)

    results[variant]['ts']           = ts
    results[variant]['I']            = I
    results[variant]['spont_I']      = spont_I
    results[variant]['infected_IP']  = infected_IP
    results[variant]['contacted_IP'] = contacted_IP

### ABC_SMC_MNN posterior parameter estimation
variants = ['wc_8_500ms.csv']
#,'wc_1_5s.csv','wc_1_10s.csv','wc_1_20s.csv','wc_1_500ms.csv','wc_4_1s.csv','wc_4_5s.csv','wc_4_10s.csv','wc_4_20s.csv','wc_4_500ms.csv','wc_8_1s.csv','wc_8_5s.csv','wc_8_10s.csv','wc_8_20s.csv','wc_8_500ms.csv']

# iterate over variants
for variant in variants:

    print('Variant:------------------------------------------------', variant)

    actual_I = np.array(results[variant]['I'])
    n = []
    n.append(len(actual_I))
    print('n',n)
    # set variant specific variables
    N_contacted  = [len(results[variant]['contacted_IP'])]
    #dt = [1]
    dt = [(results[variant]['ts'][-1] - results[variant]['ts'][0]) / len(results[variant]['ts'])]
    print('dt',dt)
    n = []
    n.append(int((results[variant]['ts'][-1] - results[variant]['ts'][0])/dt[0] ))
    print('n', n[0])
    #n = [325]
    #dt = [(results[variant]['ts'][-1] - results[variant]['ts'][0]) / len(results[variant]['ts'])]
    #print('dt', dt)

    # Number of particles
    N  = 1000

    # Number of neighbours for covariance matrix calculations
    M = 100

    # Epsilon values for temporal data decreasing over time
    epsilon_T = [90000, 70000, 50000, 30000]        # Number of generations
    G = len(epsilon_T)

    ###############################################
    # Number of simulations for each parameter set
    nsim_model = 1
   ################################################

    #######################################################################
    # Upper bound on the number of times to try the particle to be accepted
    n_models = 1
    ########################################################################

    #  Lower and upper boundaries for priors equal to the number of parameters
    lower_bound = [0.01, 0.01, 0.01, 0.01]
    upper_bound = [0.99, 0.99, 0.99, 0.99]

    # Empty matrices to store results
    old_results = {}
    new_results = {}

    # Empty vectors to store weights for parameters
    old_weights = [0] * N
    new_weights = [0] * N

    #for the number of generations
    for g in range(G):

        #Initiate the counter of particles
        i = 0
        # While the number of accepted particles is less than N_particles
        while i < N :

            if g == 0:

                # Sample from prior distributions
                beta_star = np.random.uniform(lower_bound[0],upper_bound[0],1)[0]
                mu_star = np.random.uniform(lower_bound[1],upper_bound[1],1)[0]
                gamma1_star = np.random.uniform(lower_bound[2],upper_bound[2],1)[0]
                gamma2_star = np.random.uniform(lower_bound[3],upper_bound[3],1)[0]

#                if mu_star + gamma1_star > 1:
#                    gamma1_star = 1 - mu_star

            else:

                #Select index from 1 to N from previous generation with probabilities = weights
                a = np.array(list(range(0, N)))
                p = np.random.choice(a, 1, p = old_weights)

                #get variance for this particle
                sigma = Sigma[p[0]]
                #mean is equal to sampled particle
                mean = old_results[p[0]]

                #mean 1*5, sigma 5*5
                params = sample_truncated_multivariate_normal_for_kernel(old_results[p[0]], sigma, lower_bound, upper_bound)

                beta_star = params[0][0]
                mu_star = params[0][1]
                gamma1_star = params[0][2]
                gamma2_star = params[0][3]

#                if mu_star + gamma1_star > 1:
#                    gamma1_star = 1 - mu_star

            if prior_non_zero([beta_star, mu_star, gamma1_star, gamma2_star], lower_bound, upper_bound) !=0 :

                # Set number of accepted simulations to zero
                m = 0
                distances = []

                for j in range(n_models):

                    R0 = [0]
                    I0 = [1]

                    beta_star_lst = []
                    beta_star_lst.append(beta_star)
                    mu_star_lst = []
                    mu_star_lst.append(mu_star)
                    gamma1_star_lst = []
                    gamma1_star_lst.append(gamma1_star)
                    gamma2_star_lst = []
                    gamma2_star_lst.append(gamma2_star)

                    param_grid = ParameterGrid({'beta'  : beta_star_lst,
                                                'mu'    : mu_star_lst,
                                                'gamma1': gamma1_star_lst,
                                                'gamma2': gamma2_star_lst,
                                                'N_contacted' : N_contacted,
                                                'dt'    : dt,
                                                'n'     : n,
                                                'R0'    : R0,
                                                'I0'    : I0})

                    if g == G - 1 and i %10 == 0:
                        D_star = [simulation(param, nsim_model, actual_I, False) for param in param_grid]
                    else:
                        D_star = [simulation(param, nsim_model, actual_I, False) for param in param_grid]

                    # Calculate distances
                    distance = calculate_distance_between_models(D_star[0], actual_I)

                    #print('distance',distance)
                    distances.append(distance)

                    if distance <= epsilon_T[g]:
                        m = m + 1

                if m > 0:
                                            
                    # Store results
                    new_results[i] = [beta_star, mu_star, gamma1_star, gamma2_star]

                    # Calculate weights
                    w1 = 1

                    for l in range(4):
                        pdf = uniform.pdf(x = new_results[i][l], loc = lower_bound[l], scale = upper_bound[l])
                        w1 *= pdf

                    if g == 0:
                        w2 = 1

                    else:
                        w2 = 0
                        for k in range(N):
                            pdf_w2 = prob_trunc_norm(new_results[i], mean = old_results[k], sigma = sigma, lower_bound = lower_bound, upper_bound = upper_bound)
                            w2 += old_weights[k] * pdf_w2

                    new_weight = (m/n_models)*w1/w2
                    new_weights[i] = new_weight

                    i = i + 1

                    print('Generation: -----------------------------------------------------------------', g)
                    print('Particle: -------------------------------------------------------------------', i)
        Sigma = {}

        # each particle has associated sigma for the next step from M nearest neighbors
        for p in range(N):
            Sigma[p] = get_sigma_neighbors(M, new_results[p], new_results)
            s = Sigma[p]

        old_results = new_results
        old_weights = new_weights/sum(new_weights)

    betas = []
    mus = []
    gammas1 = []
    gammas2 = []

    for i in range(N):
        betas.append(new_results[i][0])
        mus.append(new_results[i][1])
        gammas1.append(new_results[i][2])
        gammas2.append(new_results[i][3])

    beta_mean = np.mean(np.array(betas))
    mu_mean = np.mean(np.array(mus))
    gamma1_mean = np.mean(np.array(gammas1))
    gamma2_mean = np.mean(np.array(gammas2))

    print('beta mean', beta_mean)
    print('mu mean', mu_mean)
    print('gamma1 mean', gamma1_mean)
    print('gamma2 mean', gamma2_mean)

    beta_std = np.std(np.array(betas))
    mu_std = np.std(np.array(mus))
    gamma1_std = np.std(np.array(gammas1))
    gamma2_std = np.std(np.array(gammas2))

    print('beta std', beta_std)
    print('mu std', mu_std)
    print('gamma1 std', gamma1_std)
    print('gamma2 std', gamma2_std)

