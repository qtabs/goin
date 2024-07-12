import sys
sys.path.append('../')
import coin
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def unpickle_results(results_pickle):

	with open(results_pickle, 'rb') as f:
		savedict = pickle.load(f)

	mse_leak  = savedict['mse_leak'] 
	logp_leak = savedict['logp_leak'] 
	best_t    = savedict['best_t'] 
	mse_coin  = savedict['mse_coin'] 
	logp_coin = savedict['logp_coin'] 

	return(mse_leak, logp_leak, best_t, mse_coin, logp_coin)


def compute_parscan(results_pickle, parscan_config, n_samples, n_trials):

	n_par_vals = tuple([len(parscan_config[p]) for p in parscan_config.keys()])

	# Recovering previously computed results or initialisating storing variables
	if os.path.exists(results_pickle):
		mse_leak, logp_leak, best_t, mse_coin, logp_coin = unpickle_results(results_pickle)
	else:
		mse_leak  = np.nan * np.ones(n_par_vals + (n_samples,))
		logp_leak = np.nan * np.ones(n_par_vals + (n_samples,))
		best_t    = np.nan * np.ones(n_par_vals)
		mse_coin  = np.nan * np.ones(n_par_vals + (n_samples,))
		logp_coin = np.nan * np.ones(n_par_vals + (n_samples,))

	# Listing indices for all possible parameter combinations
	Ns = [(n0, n1, n2, n3) for n0 in range(n_par_vals[0]) for n1 in range(n_par_vals[1]) 
						   for n2 in range(n_par_vals[2]) for n3 in range(n_par_vals[3])]

	# Filtering all parameter combinations that have not been yet computed
	if not np.isnan(best_t).all():
		print(f'Recovered {len(Ns)-(np.isnan(best_t)).sum()} iterations from {results_pickle}')
	Ns = [N for N in Ns if np.isnan(best_t[N])]

	eng = coin.initialise_matlab_engine()
	tt0 = time.time()

	for k, N in enumerate(Ns):

		print(f'[{k+1:>3}/{len(Ns)}]', end=' ', flush=True)
		tt   = time.time()

		# Instantiate GM with current parset
		new_pars = dict([(p, parscan_config[p][N[i]]) for i, p in enumerate(parscan_config.keys())])
		new_pars['gamma_t'] = 0.25
		parsetname = '_'.join([f'{p}-{1000* new_pars[p]:03.0f}' for p in new_pars])
		pars = coin.load_pars('validation')
		pars.update(new_pars)
		gm = coin.CRFGenerativeModel({'pars': pars, 'name': parsetname})

		# Sample observations from GM
		X = gm.generate_batch(n_trials, n_samples)[0][..., 0]

		# Measuring best leaky integrator performance 
		print('Computing Leaky Integrator...', end=' ', flush=True)
		t0 = time.time()
		best_t[N] = gm.fit_best_tau(n_trials, 10 * n_samples)
		z_leak, logp_leak_all = gm.estimate_leaky_average(X)[:2]
		mse_leak[N]  = ((z_leak - X)**2).mean(1)
		logp_leak[N] = logp_leak_all.sum(1) 
		print(f'[{(time.time()-t0)/60:.1f}min]', end=' ', flush=True)

		# Measuring coin performance 
		t0 = time.time()
		print('Computing COIN...', end=' ', flush=True)
		z_coin, logp_coin_all = gm.estimate_coin(X, eng)[:2]
		mse_coin[N]  = ((z_coin - X)**2).mean(1)
		logp_coin[N] = logp_coin_all.sum(1) 
		print(f'[{(time.time()-t0)/60:.1f}min]', end=' ', flush=True)
		print(f' [it: {(time.time() - tt)/60:.1f}min]', end = ' ')
		lapsed = (time.time()-tt0) / 3600
		print(f'Total: {lapsed:.1f}h; left:{(lapsed/(k+1))*(len(Ns)-k-1):.0f}h', flush=True)

		with open(results_pickle, 'wb') as f:
			pickle.dump({'mse_leak': mse_leak, 'logp_leak': logp_leak, 'best_t': best_t, 
						 'mse_coin': mse_coin, 'logp_coin': logp_coin}, f)


def plot_results(mean, sem, parscan_config, figname=None):

	n_par_vals = tuple([len(parscan_config[p]) for p in parscan_config.keys()])
	par_names = list(parscan_config.keys())
	n = [0 for _ in n_par_vals]

	vmin, vmax  = 0.1 * np.floor(10 * mean.min()), 0.1 * np.ceil(10 * mean.max())

	fig, axs = plt.subplots(*n_par_vals[:2])
	for n[0] in range(n_par_vals[0]):
		for n[1] in range(n_par_vals[1]):
			if sem is None:
				an = [[f'{mean[n[0],n[1],n2,n3]:.2g}' for n3 in range(n_par_vals[3])] 
																for n2 in range(n_par_vals[2])]
			else:
				an = [[f'{mean[n[0],n[1],n2,n3]:.2f} \u00B1 {sem[n[0],n[1],n2,n3]:.2f}' 
																for n3 in range(n_par_vals[3])] 
																for n2 in range(n_par_vals[2])]
			yl = [f'{v:.2f}' for v in parscan_config[par_names[2]]]
			xl = [f'{v:.2f}' for v in parscan_config[par_names[3]]]
			ti = '; '.join([f'{par_names[i]} = {parscan_config[par_names[i]][n[i]]:.2f}' for i in [0,1]])

			sns.heatmap(mean[n[0], n[1], :, :], vmin=vmin, vmax=vmax, annot=an, fmt='', 
												xticklabels=xl, yticklabels=yl, ax=axs[n[0], n[1]])
			axs[n[0], n[1]].set_ylabel(par_names[2])
			axs[n[0], n[1]].set_xlabel(par_names[3])
			axs[n[0], n[1]].set_title(ti)

	fig.subplots_adjust(left=0.04,right=0.97,bottom=0.05,top=0.95,wspace=0.12,hspace=0.7)
	if figname is None:
		plt.show()
	else:
		fig.set_size_inches(40, 20)
		plt.savefig(f'{figname}.png')


results_pickle = './parscan.pickle'

# config must have four parameters
parscan_config =  {'rho_t':   np.array([0.20, 0.40, 0.60, 0.80, 0.99]),
				   'alpha_t': np.array([0.1, 0.5, 1.0, 5.0, 10.0]),
		           'mu_a':    np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
		           'si_d':    np.array([0.005, 0.01, 0.05, 0.1, 0.5])}

n_samples, n_trials = 256, 1000

compute_parscan(results_pickle, parscan_config, n_samples, n_trials)

mse_leak, logp_leak, best_t, mse_coin, logp_coin = unpickle_results(results_pickle)
mse_rat_avg = (mse_leak / mse_coin).mean(4)
mse_rat_sem = (mse_leak / mse_coin).std(4) / np.sqrt(n_samples) 
plot_results(mse_rat_avg, mse_rat_sem, parscan_config, figname='mse_rat')
