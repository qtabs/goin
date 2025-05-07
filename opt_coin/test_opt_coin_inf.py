import numpy as np
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from goin import coin as coin

import pickle

from tqdm import tqdm
import multiprocessing

import warnings
import argparse

def unpickle_results(results_pickle):

    with open(results_pickle, 'rb') as f:
       savedict = pickle.load(f)
    
    return savedict


def load_and_compare(filename):
    
    data = unpickle_results(filename)
    
    # Aggregate data
    for config_k in data.keys():
        pass
    


def compute_logpc(C, lamb=None, e_beta=None):
    """Get the log of the probability of true context (C) in contexts probabilities distribution (lamb)"""

    n_samples, n_trials = C.shape[0], C.shape[1]

    if lamb is not None:
        logp_c = np.array([[np.log(lamb[b,C[b,t],t]) for t in range(n_trials)] for b in range(n_samples)])
    elif e_beta is not None: 
        logp_c = np.array([[np.log(e_beta[C[b,t]]) for t in range(n_trials)] for b in range(n_samples)])
    return logp_c

     

def run_multiple_config(filename, config_values, n_samples, n_trials, nruns, mode='matlab', max_cores=1, max_cores_configs=1):
    
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

    if max_cores_configs is None or max_cores_configs >=1:
        # Create all config combinations
        n_par_vals = tuple(len(config_values[p]) for p in config_values.keys())
        configs = [(i, config) for i, config in enumerate(
            [(n0, n1, n2) for n0 in range(n_par_vals[0]) for n1 in range(n_par_vals[1]) for n2 in range(n_par_vals[2])]
        )]

        # Prepare input for each config
        inputs = [
            (k, config, config_values, eng, genmodel_func, max_cores, mode, n_samples, n_trials, nruns)
            for k, config in configs
        ]

        with multiprocessing.Pool(processes=max_cores) as pool:
            # tqdm needs manual handling with Pool
            for k, data_config in tqdm(pool.starmap(run_single_config, inputs), total=len(inputs)):
                data[k] = data_config

    else:
        for k, config in tqdm(enumerate(configs)):

            data_config = run_single_config(config, config_values, eng, genmodel_func, max_cores, mode, n_samples, n_trials,
                              nruns)
            data[k] = data_config

    # Save the dictionary as a pickle file
    # Save distributions of samples' logp values (avged over timepoints, as done above) for each config
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def run_single_config(config, config_values, eng, genmodel_func, max_cores, mode, n_samples, n_trials, nruns):
    # Select and instantiate list of params according to selected config
    new_pars = dict([(p, config_values[p][config[i]]) for i, p in enumerate(config_values.keys())])
    parsetname = '_'.join([f'{p}-{1000 * new_pars[p]:03.0f}' for p in new_pars])
    pars = coin.load_pars('validation')
    pars.update(new_pars)

    # Instantiate GM with current config's parset
    gm = genmodel_func({'pars': pars, 'name': parsetname})

    # Generate (sample) observations (state observations, sensory cues) and contexts with generative model
    Y, Q, C = gm.generate_batch(n_trials, n_samples)
    Y, Q, C = Y[..., 0], Q[..., 0], C[..., 0]  # last dimension is Pytorch-related, get rid of it

    # Get number of context and modify the max_contexts parameter for the COIN model
    n_ctx = len(np.unique(C))

    # Evaluate empirical beta
    e_beta = gm.empirical_expected_beta(n_samples=n_samples, n_trials=n_trials, n_contexts=max(n_ctx, 10))

    # Leaky integrator inference
    t0 = time.time()

    # should fit best tau before calling estimate_leaky_average:
    gm.fit_best_tau(n_trials, 10 * n_samples)
    z_LI, logp_LI, _ = gm.estimate_leaky_average_parallel(Y)
    mse_LI = ((z_LI - Y)**2).mean(1)
    logp_c_LI = compute_logpc(C, lamb=None, e_beta=e_beta)
    t_LI = (time.time() - t0) / 60

    # COIN inference, in Matlab and Python respectively
    if mode == 'matlab':
        t0 = time.time()
        z_coin_M, logp_coin_M, _, lamb_M = gm.estimate_coin(Y, mode='matlab', eng=eng, nruns=nruns,
                                                            n_ctx=max(n_ctx, 10), max_cores=max_cores)
        mse_coin_M = ((z_coin_M - Y)**2).mean(1)
        logp_c_coin_M = compute_logpc(C, lamb_M)
        t_M = (time.time() - t0) / 60

    # Python
    t0 = time.time()
    z_coin, logp_coin, _, lamb = gm.estimate_coin(Y, mode='python', nruns=nruns, n_ctx=max(n_ctx, 10),
                                                  max_cores=max_cores)
    mse_coin = ((z_coin - Y)**2).mean(1)
    logp_c_coin = compute_logpc(C, lamb)
    t = (time.time() - t0) / 60

    # Store
    # Note: average over time points: get one value per sample
    data_config = {'gamma_t': new_pars['gamma_t'],
                   'alpha_t': new_pars['alpha_t'],
                   'rho_t': new_pars['rho_t'],
                   'ctx_count': len(np.unique(C)),
                   'Python': {'time': t,
                              'logp_y_avg': logp_coin.mean(1),
                              'logp_y_sum': logp_coin.sum(1),
                              'mse_y_avg': mse_coin,
                              'logp_c_avg': logp_c_coin.mean(1),
                              'logp_c_sum': logp_c_coin.sum(1)},
                   'Leaky': {'time': t_LI,
                             'logp_y_avg': logp_LI.mean(1),
                             'logp_y_sum': logp_LI.sum(1),
                             'mse_y_avg': mse_LI,
                             'logp_c_avg': logp_c_LI.mean(1),
                             'logp_c_sum': logp_c_LI.sum(1)}}
    if mode == 'matlab':
        data_config['Matlab'] = {'time': t_M,
                                 'logp_y_avg': logp_coin_M.mean(1),
                                 'logp_y_sum': logp_coin_M.sum(1),
                                 'mse_y_avg': mse_coin_M,
                                 'logp_c_avg': logp_c_coin_M.mean(1),
                                 'logp_c_sum': logp_c_coin_M.sum(1)}

    return data_config


def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output', help='output file name', type=str, default="inference_results")
    parser.add_argument('-s', '--samples', help='number of samples', type=int, default=512)
    parser.add_argument('-t', '--trials', help='number of trials', type=int, default=512)
    args = parser.parse_args()

    filename = args.output
    n_samples = args.samples
    n_trials = args.trials

    warnings.filterwarnings('ignore')
    
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
    # n_samples, n_trials = 512, 512 # n_samples: samples from generative model; n_trials: timepoints (512, 512 or more)
    n_samples, n_trials = 10, 5 # n_samples: samples from generative model; n_trials: timepoints (512, 512 or more)

    # Inference runs    
    nruns = 1
    
    # 'matlab': matlab and python, 'python': only Python
    mode = 'python'
    
    # Define number of cores used across runs
    max_cores = None # But doesn't have impact

    # Define number of cores used across multiple configs
    max_cores_configs = None

    # Results path
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")):
        os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), "output"))
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", f"{filename}.pkl")
    run_multiple_config(filepath, config_values=config_values, n_samples=n_samples, n_trials=n_trials, nruns=nruns, mode=mode, max_cores=max_cores, max_cores_configs=max_cores_configs)

    # Verify that the file has been saved
    if os.path.exists(filepath): print(f"Results saved in {filepath}")
    else: print("Results file not found")   


    # Later: find optimal hyperparam configuration and run comparison with it



if __name__=="__main__":
    # multiprocessing.set_start_method('fork')
    main()