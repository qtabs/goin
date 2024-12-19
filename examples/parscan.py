import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import coin
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import codecs

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())


def unpickle_results(results_pickle):

	with open(results_pickle, 'rb') as f:
		savedict = pickle.load(f)

	mse_leak  = savedict['mse_leak'] 
	logp_leak = savedict['logp_leak'] 
	best_t    = savedict['best_t'] 
	mse_coin  = savedict['mse_coin'] 
	logp_coin = savedict['logp_coin'] 
	ebeta_ctx = savedict['ebeta_ctx']
	coin_ctx  = savedict['coin_ctx']

	return(mse_leak, logp_leak, best_t, mse_coin, logp_coin, ebeta_ctx, coin_ctx)


def compute_parscan(genmodel_func, results_pickle, parscan_config, n_samples, n_trials):

	n_par_vals = tuple([len(parscan_config[p]) for p in parscan_config.keys()])

	# Recovering previously computed results or initialisating storing variables
	if os.path.exists(results_pickle):
		mse_leak, logp_leak, best_t, mse_coin, logp_coin, ebeta_ctx, coin_ctx = unpickle_results(results_pickle)
	else:
		mse_leak  = np.nan * np.ones(n_par_vals + (n_samples,))
		logp_leak = np.nan * np.ones(n_par_vals + (n_samples,))
		best_t    = np.nan * np.ones(n_par_vals)
		mse_coin  = np.nan * np.ones(n_par_vals + (n_samples,))
		logp_coin = np.nan * np.ones(n_par_vals + (n_samples,))
		ebeta_ctx = np.nan * np.ones(n_par_vals + (n_samples,))
		coin_ctx  = np.nan * np.ones(n_par_vals + (n_samples,))

	# Listing indices for all possible parameter combinations
	Ns = [(n0, n1, n2, n3) for n0 in range(n_par_vals[0]) for n1 in range(n_par_vals[1]) 
						   for n2 in range(n_par_vals[2]) for n3 in range(n_par_vals[3])]

	# Filtering all parameter combinations that have not been yet computed
	if not np.isnan(best_t).all():
		print(f'Recovered {len(Ns)-(np.isnan(best_t)).sum()} iterations from {results_pickle}')
	
	Ns = [N for N in Ns if np.isnan(best_t[N])]

	Ns = [(2, 4, 4, 4)]

	if Ns == []:
		return


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
		# gm = coin.CRFGenerativeModel({'pars': pars, 'name': parsetname})
		# gm = coin.UrnGenerativeModel({'pars': pars, 'name': parsetname})
		gm = genmodel_func({'pars': pars, 'name': parsetname})

		print('Computing ', end=' ', flush=True)

		# Sample observations from GM

		X, _, C = gm.generate_batch(n_trials, n_samples)
		X, C = X[..., 0], C[..., 0]

		# Measuring baseline performance 
		print('[Baseline ', end=' ', flush=True)
		t0 = time.time()
		best_t[N] = gm.fit_best_tau(n_trials, 10 * n_samples)
		z_leak, logp_leak_all = gm.estimate_leaky_average(X)[:2]
		mse_leak[N]  = ((z_leak - X)**2).mean(1) # Global mean value over n_samples and n_trials?
		logp_leak[N] = logp_leak_all.sum(1)  # .sum(1)? # Reason to delete axis 1: code expects log to be n_samples, n_trials ("same dimensional arrangement as y") but runCOIN.m script specifies logp as one-dim n_samples-long
		e_beta       = gm.empirical_expected_beta(n_trials=n_trials)
		logp_ebeta   = [[np.log(e_beta[C[b, t]]) for t in range(n_trials)] for b in range(n_samples)]
		ebeta_ctx[N] = np.array(logp_ebeta).mean(1)
		print(f'{(time.time()-t0)/60:.1f}min]', end=' ', flush=True)

		# Measuring coin performance 
		t0 = time.time()
		print('[COIN ', end=' ', flush=True)
		z_coin, logp_coin_all, _, lamb = gm.estimate_coin(X, eng)
		mse_coin[N]   = ((z_coin - X)**2).mean(1)
		logp_coin[N]  = logp_coin_all.sum(1)  # .sum(1)? # Reason to delete axis 1: code expects log to be n_samples, n_trials ("same dimensional arrangement as y") but runCOIN.m script specifies logp as one-dim n_samples-long
		logp_ctx_coin = [[np.log(lamb[b,C[b,t],t]) for t in range(n_trials)] for b in range(n_samples)]
		coin_ctx[N]   = np.array(logp_ctx_coin).mean(1)
		print(f'{(time.time()-t0)/60:.1f}min]', end=' ', flush=True)
		print(f'[it: {(time.time() - tt)/60:.1f}min]', end = ' ')
		print(f'[mse_rat={(mse_leak[N]/mse_coin[N]).mean():.1f},', end = ' ')
		print(f'logp_dif={(coin_ctx[N] - ebeta_ctx[N]).mean():.1f}] ', end = ' ')
		lapsed = (time.time()-tt0) / 3600
		print(f'Total: {lapsed:.1f}h; left:{(lapsed/(k+1))*(len(Ns)-k-1):.0f}h', flush=True)
		with open(results_pickle, 'wb') as f:
			pickle.dump({'mse_leak': mse_leak,   'logp_leak': logp_leak, 'best_t': best_t, 
						 'mse_coin': mse_coin,   'logp_coin': logp_coin,
						 'ebeta_ctx': ebeta_ctx, 'coin_ctx':  coin_ctx}, f)


def recompute_nan_trials(results_pickle, parscan_config, n_trials):

	n_par_vals = tuple([len(parscan_config[p]) for p in parscan_config.keys()])
	mse_leak, logp_leak, best_t, mse_coin, logp_coin, ebeta_ctx, coin_ctx = unpickle_results(results_pickle)
	
	# Listing indices for all possible parameter combinations
	Ns = [(n0, n1, n2, n3) for n0 in range(n_par_vals[0]) for n1 in range(n_par_vals[1]) 
						   for n2 in range(n_par_vals[2]) for n3 in range(n_par_vals[3])]

	d = coin_ctx - ebeta_ctx
	Ns = [N for N in Ns if np.isnan(best_t[N]) or np.isinf(d[N]).any() or np.isnan(d[N]).any()]
	print(f'Found {len(Ns)} parametrisations with NaNs; {(np.isinf(d) | np.isnan(d)).sum()} trials in total')

	if Ns == []:
		return

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
		gm = coin.UrnGenerativeModel({'pars': pars, 'name': parsetname}) # TODO: replace with more optimal

		print('Computing ', end=' ', flush=True)

		nan_samples = np.where(np.isinf(d[N]) | np.isnan(d[N]))[0]

		# Sample observations from GM
		X, _, C = gm.generate_batch(n_trials, len(nan_samples))
		X, C = X[..., 0], C[..., 0]

		# Measuring baseline performance 
		print('[Baseline ', end=' ', flush=True)
		t0 = time.time()
		gm.best_t = best_t[N]
		z_leak, logp_leak_all     = gm.estimate_leaky_average(X)[:2]
		mse_leak[N][nan_samples]  = ((z_leak - X)**2).mean(1)
		logp_leak[N][nan_samples] = logp_leak_all.sum(1) 

		e_beta     = gm.empirical_expected_beta(n_trials=n_trials)
		logp_ebeta = [[np.log(e_beta[C[b, t]]) for t in range(n_trials)] for b in range(C.shape[0])]
		ebeta_ctx[N][nan_samples] = np.array(logp_ebeta).mean(1)
		print(f'{(time.time()-t0)/60:.1f}min]', end=' ', flush=True)

		# Measuring coin performance 
		t0 = time.time()
		print('[COIN ', end=' ', flush=True)
		z_coin, logp_coin_all, _, lamb = gm.estimate_coin(X, eng)

		mse_coin[N][nan_samples]  = ((z_coin - X)**2).mean(1)
		logp_coin[N][nan_samples] = logp_coin_all.sum()  # .sum(1)
		logp_ctx_coin = [[np.log(lamb[b,C[b,t],t]) for t in range(n_trials)] for b in range(C.shape[0])]
		coin_ctx[N][nan_samples]  = np.array(logp_ctx_coin).mean(1)

		print(f'{(time.time()-t0)/60:.1f}min]', end=' ', flush=True)
		print(f'[it: {(time.time() - tt)/60:.1f}min]', end = ' ')
		print(f'[mse_rat={(mse_leak[N]/mse_coin[N]).mean():.1f},', end = ' ')
		print(f'logp_dif={(coin_ctx[N] - ebeta_ctx[N]).mean():.1f}] ', end = ' ')
		lapsed = (time.time()-tt0) / 3600
		print(f'Total: {lapsed:.1f}h; left:{(lapsed/(k+1))*(len(Ns)-k-1):.0f}h', flush=True)
		with open(results_pickle, 'wb') as f:
			pickle.dump({'mse_leak': mse_leak,   'logp_leak': logp_leak, 'best_t': best_t, 
						 'mse_coin': mse_coin,   'logp_coin': logp_coin,
						 'ebeta_ctx': ebeta_ctx, 'coin_ctx':  coin_ctx}, f)


def plot_results(mean, sem, parscan_config, scale=0.1, figname=None):

	n_par_vals = tuple([len(parscan_config[p]) for p in parscan_config.keys()])
	par_names = list(parscan_config.keys())
	n = [0 for _ in n_par_vals]

	vmin, vmax  = scale * np.floor(1/scale * mean.min()), scale * np.ceil(1/scale * mean.max())

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



if __name__ == '__main__':
    
    
	# config must have four parameters
	parscan_config =  {'rho_t':   np.array([0.20, 0.40, 0.60, 0.80, 0.99]),	
					'alpha_t': np.array([0.1, 0.5, 1.0, 5.0, 10.0]),
					'mu_a':    np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
					'si_d':    np.array([0.005, 0.01, 0.05, 0.1, 0.5])}

	n_samples, n_trials = 2, 5  # 256, 1000
 
	# Compare gen model formulations
	genmodel_variants = {'ExplicitGenerativeModel': coin.ExplicitGenerativeModel, 
                      'CRFGenerativeModel': coin.CRFGenerativeModel,
                      'UrnGenerativeModel': coin.UrnGenerativeModel}
	
	for variant in genmodel_variants:
		print("Gen. model variant: ", variant)		
  
		genmodel_func = genmodel_variants[variant]
		
		results_pickle = f'./parscan_{variant}.pickle' # TODO: diff results & benchmarks

		compute_parscan(genmodel_func, results_pickle, parscan_config, n_samples, n_trials)
		recompute_nan_trials(results_pickle, parscan_config, n_trials)

		mse_leak, logp_leak, best_t, mse_coin, logp_coin, ebeta_ctx, coin_ctx = unpickle_results(results_pickle)

		mse_rat_avg = (mse_leak / mse_coin).mean(4)
		mse_rat_sem = (mse_leak / mse_coin).std(4) 
		plot_results(mse_rat_avg, mse_rat_sem, parscan_config, scale=0.1, figname='mse_rat')

		ctx_diff_avg = (coin_ctx - ebeta_ctx).mean(4)
		ctx_diff_sem = (coin_ctx - ebeta_ctx).std(4)
		plot_results(ctx_diff_avg, ctx_diff_sem, parscan_config, scale=0.01, figname='ctx_logp_dif')
  
		


