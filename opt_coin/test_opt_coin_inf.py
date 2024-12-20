import numpy as np
import matplotlib.pyplot as plt
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from COIN_Python import coin as coinp
from goin import coin as coin

import pandas as pd
import pickle

def unpickle_results(results_pickle):

    with open(results_pickle, 'rb') as f:
       savedict = pickle.load(f)
    
    return savedict


def load_and_compare(filename):
    
    data = unpickle_results(filename)
    
    # Aggregate data
    for config_k in data.keys():
        pass
    


def compute_logpc(C, lamb=None):
    """Get probability of true context in contexts probabilities distribution and compute log"""
    
    if lamb is not None:
        n_samples, n_trials = C.shape[0], C.shape[1]
        logp_c = np.array([[np.log(lamb[b,C[b,t],t]) for t in range(n_trials)] for b in range(n_samples)])
    else:
            logp_c = np.log(C[:, :] == 0)
    return logp_c


def store_inf_results(filename, C, logp, lamb, t, variant, k, new_pars, version):
    """Update df_results with performance and metainformation from an inference estimation"""
    
    
    if lamb is not None:
        n_samples, n_trials = C.shape[0], C.shape[1]
        logp_c = np.array([[np.log(lamb[batch,C[batch,t],t]) for t in range(n_trials)] for batch in range(n_samples)])
    else:
        logp_c = np.log(C[:, :] == 0)
        
    data = {
        'model_version': version,
        'model_variant': variant,
        'config_id': k,
        'rho_t': new_pars['rho_t'],
        'alpha_t': new_pars['alpha_t'],
        'gamma_t': new_pars['gamma_t'],
        'time': t,
        'logp_y': logp,
        'logp_c': logp_c,
    }
    
    # Save the dictionary as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
     
    
    
def run_multiple_config(filename, config_values, n_samples, n_trials, nruns, mode='matlab'):   
    
    # If Matlab, activate matlab engine        
    eng = None
    if mode == 'matlab':
        eng = coin.initialise_matlab_engine()

    # Define all combinations of hyperparameters values provided in config_values
    n_par_vals = tuple([len(config_values[p]) for p in config_values.keys()])
    configs = [(n0, n1, n2) for n0 in range(n_par_vals[0]) for n1 in range(n_par_vals[1]) 
						   for n2 in range(n_par_vals[2])]
    
        
    # Store inference predictions: predicted observations, log proba of input sequence, conetext responsibilities 
    data = {}
    
    # Select model variant
    genmodel_func = coin.UrnGenerativeModel
        
                
    # For each config, run nruns times and get average perf per Matlab or Python
    for k, config in enumerate(configs):          
        
        # Select and instantiate list of params according to selected config
        new_pars = dict([(p, config_values[p][config[i]]) for i, p in enumerate(config_values.keys())])
        parsetname = '_'.join([f'{p}-{1000* new_pars[p]:03.0f}' for p in new_pars])
        pars = coin.load_pars('validation')
        pars.update(new_pars)
        
        
        # Instantiate GM with current config's parset
        gm = genmodel_func({'pars': pars, 'name': parsetname})            
        
        
        # Generate (sample) observations (state observations, sensory cues) and contexts with generative model
        Y, Q, C = gm.generate_batch(n_trials, n_samples)
        Y, Q, C = Y[..., 0], Q[..., 0], C[..., 0] # last dimension is Pytorch-related, get rid of it
        
            
        # Leaky integrator inference
        t0 = time.time()            
        z_LI, logp_LI, _= gm.estimate_leaky_average_parallel(Y)
        logp_c_LI = compute_logpc(C, lamb=None)
        t_LI = (time.time()-t0)/60

        # COIN inference, in Matlab and Python respectively
        if mode=='matlab':
            t0 = time.time()
            z_coin_M, logp_coin_M, _, lamb_M = gm.estimate_coin(Y, mode='matlab', eng=eng, nruns=nruns)
            logp_c_coin_M = compute_logpc(C, lamb_M)
            t_M = (time.time()-t0)/60

        # Python
        t0 = time.time()            
        z_coin, logp_coin, _, lamb = gm.estimate_coin(Y, mode='python', nruns=nruns)
        logp_c_coin = compute_logpc(C, lamb)
        t = (time.time()-t0)/60
                
        
        # Store
        # Note: average over samples or time points
        data_config = {'gamma_t': new_pars['gamma_t'],
                        'alpha_t': new_pars['alpha_t'],
                        'rho_t': new_pars['rho_t'],
                        'Python': {'time': t,
                                   'logp_y_avg':logp_coin.mean(1),
                                   'logp_y_sum':logp_coin.sum(1),
                                   'logp_c_avg': logp_c_coin.mean(1),
                                   'logp_c_sum': logp_c_coin.sum(1)},
                        'Leaky': {'time': t_LI,
                                  'logp_y_avg': logp_LI.mean(1),
                                  'logp_y_sum': logp_LI.sum(1),
                                  'logp_c_avg': logp_c_LI.mean(1),
                                  'logp_c_sum': logp_c_LI.sum(1)}}
        
        if mode == 'matlab':
            data_config['Matlab'] = {'time': t_M,
                                     'logp_y_avg': logp_coin_M.mean(1),
                                     'logp_y_sum': logp_coin_M.sum(1),
                                     'logp_c_avg': logp_c_coin_M.mean(1),
                                     'logp_c_sum': logp_c_coin_M.sum(1)}

        
        data[k] = data_config
        
        
        
    # Save the dictionary as a pickle file
    # Saving the distributions of logp values (n_samples, n_trials) to keep time points information
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        

    
def main():
    
    # Apply Heald's inference
    # z_coin : np.array
    #     predictions hat{y}_t for the observations y_{1:t-1}; same dimensional arrangement as y (i.e.: n_samples, n_trials)
    # logp   : np.array
    #     log-probabilities of the input sequence y_t under the COIN posterior distribution; same
    #     dimensional arrangement as y
    # lamb   : np.array
    #     responsibilities lambda^c_t for each context c and time-step t. dim 0 runs across 
    #     batches, dim 1 across time points, dim 2 across contexts (dimension equals the maximum
    #     number of contexts of the COIN model, currently set to 10+1) # TODO: make sure no problem in inverting dimensions 1 and 2
    
    # Metrics (log)
    # p(c=c_t|y_1:t-1)
    # p(y=y_t|y_1:t-1)
    
    # Set configuration params: define possible parameter values
    config_values =  {'rho_t':   np.array([0.10, 0.50, 0.90]),	
					'alpha_t': np.array([0.1, 1.0, 10.0]),
					'gamma_t':    np.array([0.1, 1.0, 10.0])}
    # config_values =  {'rho_t':   np.array([0.10, 0.90]),	
	# 				'alpha_t': np.array([0.1, 10.0]),
	# 				'gamma_t':    np.array([0.1, 10.0])}
    
    # Generative model sample draws
    n_samples, n_trials = 512, 512 # n_samples: samples from generative model; n_trials: timepoints (512, 512 or more)

    # Inference runs    
    nruns = 1
    
    # 'matlab': matlab and python, 'python': only Python
    mode = 'matlab'
    
    # Results path
    filename = "comparison_results.pkl"
    
    run_multiple_config(filename, config_values=config_values, n_samples=n_samples, n_trials=n_trials, nruns=nruns, mode=mode)
    # load_and_compare(filename)   
    
    # Later: find optimal hyperparam configuration and run comparison with it



if __name__=="__main__":
    main()