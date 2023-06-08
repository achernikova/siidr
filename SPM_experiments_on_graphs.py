import igraph as ig
import pandas as pd
from igraph import *
import random
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs

import SPM_graph_simulation as spm

graphs = []

#graphs_filenames = ["erdos_renyi_lambda11.net", "erdos_renyi_lambda100.net", "erdos_renyi_lambda500.net", "erdos_renyi_lambda999.net"]
#graphs_filenames = ["barabasi-albert_lambda6.net", "barabasi-albert_lambda35.net", "barabasi-albert_lambda130.net", "barabasi-albert_lambda222.net", "barabasi-albert_lambda508.net"]
graphs_filenames = ["configuration_model_lambda9.net", "scale_free_lambda22.net", "email_EU_lambda103.net", "facebook_lambda162.net", "oregon_lambda60.net"]
#graphs_filenames = ["watts_strogatz_lambda2.net","watts_strogatz_lambda10.net","watts_strogatz_lambda100.net","watts_strogatz_lambda500.net","watts_strogatz_lambda999.net"]
#graphs_filenames = ["git149.net", "twitter209.net", "bitcoin54.net"]

for g_filename in graphs_filenames:
    g = Graph.Read_Pajek("graphs/"+g_filename)
    graphs.append(g)

#g_names = ["er11", "er100", "er500", "er999"]
#g_names = ["ba35", "ba130", "ba222", "ba508"]
g_names = ["cm9", "sf22", "emailEU103", "facebook162", "oregon60"]
#g_names =["ws10","ws100","ws500","ws999"]
#g_names = ["git149","twitter209","bitcoin54"]
seed = 1
random.seed(seed)
np.random.seed(seed)

siidr_params = {}
#mu = 0.01
#mu = 0.5
mu = 0.5
s = [0.1, 0.3, 0.5, 0.7, 1, 1.3, 1.5, 1.7, 2]
gamma1 = 0.5
gamma2 = 0.5

num_experiments = 50 # number of experiments, each having a different starting point
num_runs = 100 # we do num_runs for each starting point
num_steps = 1000  # number of simulation steps
frac_infected = None  # a single starting point (default), not a fraction
lcc_only = False

model = "SIIDR"
#model = "SIR"

for i in range(len(g_names)):

    g = graphs[i]
    g_name = g_names[i]

    A = g.get_adjacency_sparse()
    vals, vecs = eigs(A.asfptype())
    eigenv_max = max(vals)

    for i in range(len(s)):
        beta = s[i] * mu / eigenv_max
        results_dir = "{}/{}/{}/".format(g_name, s[i],model)
        spm.run_spm(g, results_dir, frac_infected, lcc_only, model, num_experiments, num_runs, num_steps, beta, mu, gamma1, gamma2, alpha=0)



