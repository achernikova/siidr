import numpy as np
import pandas as pd
import igraph as ig
from igraph import *
import random
import sys
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

class Simulation:

    def __init__(self, param_grid, actual_I, k):

        # input for simulations
        self.param_grid = param_grid
        self.actual_I   = actual_I
        self.k          = k

        # results of simualtions
        self.aic     = []
        self.tot_avg = []
        self.tot_std = []
        self.tot_min = []
        self.tot_max = []

        # best parameters
        self.best_params            = OrderedDict()
        self.best_params['beta']    = []
        self.best_params['mu']      = []
        self.best_params['gamma']   = []
        self.best_params['gamma1']  = []
        self.best_params['gamma2']  = []
        self.best_params['R0']      = []
        self.best_params['tot_avg'] = []
        self.best_params['tot_std'] = []
        self.best_params['tot_min'] = []
        self.best_params['tot_max'] = []
        self.best_params['aic']     = []

    #stichastic simulation of SIS model
    @staticmethod
    def stochastic_sis(beta, mu, N, dt, n, R0, I0):

        # lists for compartments and time
        S  = [N - I0 - R0]
        I  = [I0]
        t  = [0]

        # iterate until the end of the simulation
        for i in range(n - 1):

            # check if there are no more infected
            if I[-1] == 0 :
                for j in np.arange(i, n - 1, 1):
                    # update
                    S.append(S[-1])
                    I.append(I[-1])
                    t.append(t[-1] + dt)
                break

            new_inf  = np.random.binomial(S[-1], 1.0 - np.exp(- beta * I[-1] / N * dt))

            # total individual leaving I
            new_susc = np.random.binomial(I[-1], 1.0 - np.exp(- mu * dt))

            # update
            S.append(S[-1]   - new_inf + new_susc)
            I.append(I[-1]   + new_inf - new_susc)

        return np.array(S), np.array(I)

    #stochastic simulation of SEIR model
    @staticmethod
    def stochastic_seir(beta, mu, gamma, N, dt, n, R0, I0):

        # lists for compartments and time
        S  = [N - I0 - R0]
        I  = [I0]
        E = [0]
        R = [R0]
        t  = [0]

        # iterate until the end of the simulation
        for i in range(n - 1):

            # check if there are no more infected
            if I[-1] == 0 :
                for j in np.arange(i, n - 1, 1):
                    # update
                    S.append(S[-1])
                    E.append(E[-1])
                    I.append(I[-1])
                    R.append(R[-1])
                    t.append(t[-1] + dt)
                break

            new_exp  = np.random.binomial(S[-1], 1.0 - np.exp(- beta * I[-1] / N * dt))
            new_inf  = np.random.binomial(E[-1], 1.0 - np.exp(- gamma * dt))
            new_rec  = np.random.binomial(I[-1], 1.0 - np.exp(- mu * dt))

            # update
            S.append(S[-1]   - new_exp)
            E.append(E[-1]   + new_exp - new_inf)
            I.append(I[-1]   + new_inf - new_rec)
            R.append(R[-1]   + new_rec)

        return np.array(S),np.array(E), np.array(I), np.array(R)


    #stochastic simulation of SI (gamma1 = 0, gamma2 = 0, mu = 0)/SIR (gamma1 = 0, gamma2 = 0)/SIIDR models
    @staticmethod
    def stochastic(beta, mu, gamma1, gamma2, N, dt, n, R0, I0):
        
        # lists for compartments and time
        S  = [N - I0 - R0]
        I  = [I0]
        ID = [0]
        R  = [R0]
        t  = [0]

        # iterate until the end of the simulation
        for i in range(n - 1):

            # check if there are no more infected
            if I[-1] == 0 and ID[-1] == 0:
                for j in np.arange(i, n - 1, 1):
                    # update
                    S.append(S[-1])
                    I.append(I[-1])
                    ID.append(ID[-1])
                    R.append(R[-1])
                    t.append(t[-1] + dt)
                break

            new_inf  = np.random.binomial(S[-1], 1.0 - np.exp(-beta * I[-1] / N * dt))

            # total individual leaving I
            I_leaving = np.random.binomial(I[-1], 1.0 - np.exp(- (mu + gamma1) * dt))

            if I_leaving != 0:
                # compute relative probs of going to ID or R
                p_R  = mu     / (mu + gamma1)
                p_ID = gamma1 / (mu + gamma1)

                # sample new sleeping and recovered
                I_multi  = np.random.multinomial(I_leaving, [p_R, p_ID])
                new_rec  = I_multi[0]
                new_dorm = I_multi[1]

            else:
                new_rec  = 0
                new_dorm = 0

            new_act  = np.random.binomial(ID[-1], 1.0 - np.exp(-gamma2 * dt))

            # update
            S.append(S[-1]   - new_inf)
            I.append(I[-1]   + new_inf  + new_act - new_rec - new_dorm)
            ID.append(ID[-1] + new_dorm - new_act)
            R.append(R[-1]   + new_rec)


        # substract immunes from R
        R = np.array(R) - R0

        return np.array(S), np.array(I), np.array(ID), np.array(R)

    @staticmethod
    def AIC(y1, y2, k, n_param):

        # dimensionality control
        assert y1.shape[0] == y2.shape[0], 'y1 and y2 must have the same length'
        #residuals = y1-y2
        #res = check_residuals_for_normal_distr_normaltest(residuals)
        #if res:
        #   print('normally distributed residuals')

        s2  = np.sum((y1 - y2)**2)
        aic = n_param * np.log(s2/n_param) + 2 * k

        return aic


    def simulation(self, param, nsim, model):

        # get params
        beta   = param['beta']
        mu     = param['mu']
        gamma  = param['gamma']
        gamma1 = param['gamma1']
        gamma2 = param['gamma2']
        N      = param['N']
        dt     = param['dt']
        n      = param['n']
        R0     = param['R0']
        I0     = param['I0']
        print

        tot_I = 0

        if model == "sis":

            S, I = [], []
            for i in range(nsim):
                S_i, I_i = self.stochastic_sis(beta, mu, N, dt, n, R0, I0)
                S.append(S_i)
                I.append(I_i)

            tot_I   = np.array(I)

        elif model == "seir":

            S, E, I, R = [], [], [], []

            for i in range(nsim):
                S_i, E_i, I_i ,R_i = self.stochastic_seir(beta, mu, gamma, N, dt, n, R0, I0)
                S.append(S_i)
                E.append(E_i)
                I.append(I_i)
                R.append(R_i)

            tot_I = np.array(I) + np.array(R)


        else:

            S, I, ID, R = [], [], [], []
            for i in range(nsim):
                S_i, I_i, ID_i, R_i = self.stochastic(beta, mu, gamma1, gamma2,
                                                      N, dt, n, R0, I0)
                S.append(S_i)
                I.append(I_i)
                ID.append(ID_i)
                R.append(R_i)

            tot_I   = np.array(I) + np.array(ID) + np.array(R)

        tot_avg = tot_I.mean(axis=0)
        tot_std = tot_I.std(axis=0)
        tot_min = tot_I.min(axis=0)
        tot_max = tot_I.max(axis=0)

        # get AIC
        aic = self.AIC(self.actual_I, tot_avg, self.k, n)

        return aic, tot_avg, tot_std, tot_min, tot_max


    def simulation_over_param_grid(self, nsim, model):

        results = [self.simulation(param, nsim, model) for param in self.param_grid]

        # get aic and Is
        self.aic = np.array([results[i][0] for i in range(len(results))])
        self.tot_avg = np.array([results[i][1] for i in range(len(results))])
        self.tot_std = np.array([results[i][2] for i in range(len(results))])
        self.tot_min = np.array([results[i][3] for i in range(len(results))])
        self.tot_max = np.array([results[i][4] for i in range(len(results))])


    def get_best_models(self, tolerance = 10):

        # min aic
        aic_min = np.min(self.aic)
        idx_min = np.argmin(self.aic)

        print('min params')
        print(self.param_grid[idx_min]['beta'])
        print(self.param_grid[idx_min]['mu'])
        print(self.param_grid[idx_min]['gamma'])
        print(self.param_grid[idx_min]['gamma1'])
        print(self.param_grid[idx_min]['gamma2'])

        # iterate over AICs
        for i in range(len(self.aic)):

            # if best_AIC - tol < AIC(i) < best_AIC + tolerance keep AIC(i) and related params
            if np.abs(self.aic[i] - aic_min) < tolerance:

                self.best_params['beta'].append(self.param_grid[i]['beta'])
                self.best_params['mu'].append(self.param_grid[i]['mu'])
                self.best_params['gamma'].append(self.param_grid[i]['gamma'])
                self.best_params['gamma1'].append(self.param_grid[i]['gamma1'])
                self.best_params['gamma2'].append(self.param_grid[i]['gamma2'])
                self.best_params['R0'].append(self.param_grid[i]['R0'])
                self.best_params['tot_avg'].append(self.tot_avg[i])
                self.best_params['tot_std'].append(self.tot_std[i])
                self.best_params['tot_min'].append(self.tot_min[i])
                self.best_params['tot_max'].append(self.tot_max[i])
