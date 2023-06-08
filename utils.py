import numpy as np
import pandas as pd
import igraph as ig
from igraph import *
import random
import sys
import os
import matplotlib.pyplot as plt
from collections import OrderedDict


def check_residuals_for_normal_distr_normaltest(residuals):

    stat, p = stats.normaltest(residuals)

    print(p)

    if p < 0.05:
        return True

    else:
        return False

def import_data(variant):

    # path to input file
    path = 'wc_data/' + variant

    # import data
    df = pd.read_csv(path)
    df = df.replace('-', -1)

    # sort values by timestamp
    df.sort_values(by = 'ts', ascending = True, inplace = True)
    df.reset_index(inplace = True, drop = True)

    #convert necessary fields to numeric or string
    df["ts"] = pd.to_numeric(df["ts"])
    df["id.orig_p"] = pd.to_numeric(df["id.orig_p"])
    df["id.resp_p"] = pd.to_numeric(df["id.resp_p"])
    df["resp_bytes"] = pd.to_numeric(df["resp_bytes"])
    df['id.orig_h']  = df['id.orig_h'].astype('string')
    df['id.resp_h']  = df['id.resp_h'].astype('string')

    return df

def epi_wcry(df, cut):

    # our observations starts at the moment of the first attack
    df.drop(df.loc[df.index < df.loc[df['id.resp_p'] == 445].index[0]].index, inplace = True)
    df.reset_index(inplace = True, drop = True)
    df.ts = df.ts - df.ts[0] # starts from t=0

    # IP and infection time of patient zero
    patient_zero = df['id.orig_h'][0]
    ts_zero      = df['ts'][0]

    # trace epidemics evolution
    infected_IP  = OrderedDict()         # key: infected IP, value: ts of infection
    infected_IP[patient_zero]  = ts_zero # infection of patient zero

    contacted_IP = OrderedDict()         # key: contacted IP, value: contacts before infection
    contacted_IP[patient_zero] = 0       # contacts of patient zero

    ts      = [df['ts'][0]]              # timestamp of simulation
    I       = [1]                        # current n. of infected at each ts
    spont_I = 0                          # n. of infected IP without a previous contact

    # iterate over df
    for index, row in df.iterrows():
        ts.append(row['ts'])
        current_infected = I[-1]

        # look only at attacks towards the internal network
        if row['id.resp_h'].startswith('192.168') and row['id.orig_h'].startswith('192.168'):

            # all malicious traffic is on port 445
            if row['id.resp_p'] == 445:

                # if it is the first attack to this IP store it among the contacted
                if row['id.resp_h'] not in contacted_IP.keys():
                    contacted_IP[row['id.resp_h']] = 1
                else:
                    # if it is not yet infected go on counting contacts
                    if row['id.resp_h'] not in infected_IP.keys():
                        contacted_IP[row['id.resp_h']] += 1

                # if it is the first attack by this IP store it among the infected
                if row['id.orig_h'] not in infected_IP.keys():
                    infected_IP[row['id.orig_h']] = row['ts']
                    current_infected += 1

                    # check that the new infected has been contacted before
                    if row['id.orig_h'] not in contacted_IP.keys():
                        spont_I += 1

        I.append(current_infected)
        
    if cut:

        print('cut')

        max_value = max(I)
        max_index = I.index(max_value)
        print('max index', max_index)
        ts = ts[:max_index + 1]
        I = I[:max_index + 1]

    return ts, I, spont_I, infected_IP, contacted_IP
