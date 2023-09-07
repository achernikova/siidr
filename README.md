# Susceptible-Infected-Infected Dormant-Recovered (SIIDR) model for self-propagating malware (SPM)
Code for Modeling Self-Propagating Malware with Epidemiological Models manuscript (https://appliednetsci.springeropen.com/articles/10.1007/s41109-023-00578-z)
where we propose a new epidemiological-inspired model for SPM, called SIIDR, and extensively study its essential characteristics.

Language: python 3.9
Dependencies: scipy 1.9.3, scikit-learn 1.0.2, pandas 1.4.2, numpy 1.21.5, networkx 2.8.8, matplotlib 3.5.1, igraph

contact: chernikova.a@northeastern.edu

## Project Structure
*	graphs/
		* .net files containing graphs for threshold evaluation subsection
*	AIC_model_fitting.py
*	SMC_MNN_parameter_estimation.py
*	SPM_experiments_on_graphs.py
*	SPM_graph_simulation.py
*	SPM_stochastic_simulation.py
*	utils.py

## Stochastic Simulation of SI, SIR, SIS, SEIR, SIIDR models
SPM_stochastic_simulation.py file contains the utility functions for stochastic simulation of models. Additionally, it contains functions for simulation over the parameters grid and for selecting the best model according to AIC. 

## Fitting the WannaCry(WC) Traces and Model Selection based on the Akaike Information Criterion (AIC)
To perform fitting of WC traces and model selection, run AIC_model_fitting.py. To do this, you will need files with WC traces available on reasonable request (contact: chernikova.a@northeastern.edu).
The output will contain the minimum AIC score for each model, associated parameters, and the image with fitted curves for all WC variants and all models, along with the actual trajectory of infected nodes.
Additionally, the file contains the functionality for reading the WC traces and locating all malicious traffic towards the internal network.

## Parameters Estimation with Sequential Monte-Carlo (SMC-MNN) Approach when Covariance Matrix is Calculated using M Nearest Neighbors of the Particle

## Simulation of SIIDR Propagation on Graphs








