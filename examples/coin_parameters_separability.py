import sys
sys.path.append('../')
import coin
import pykalman
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.stats
import time
#from pathos.multiprocessing import Pool
from multiprocessing import Pool

def kalman_estimate_parameters(y, gm):

	# LGD parameters
	mu_a, si_a = gm.mu_a, gm.si_a
	mu_d, si_d = 0, gm.si_d
	si_q, si_r = gm.si_q, gm.si_r

	# We will initialise the states according to E[steady state]
	# E[x]_{a~N(mu_a, si_a),d~N(mu_d, si_d)} = E[d] / (1 - E[a]) = 0 
	ss_x  = mu_d / (1 - mu_a)
	# Assuming si_a << 1 we can approximate the covariance by:
	si2_x = si_q**2 / (1 - mu_a**2) + si_d**2 

	# Kalman filter parameters
	kalman = pykalman.KalmanFilter(em_vars=['transition_matrices'])
	kalman.initial_state_mean       = [0, 1]
	kalman.initial_state_covariance = [[si2_x, 0], [0, 0]]
	kalman.transition_matrices      = [[mu_a, mu_d], [0, 1]]
	kalman.transition_covariance    = [[si_q**2, 0], [0, 0]]
	kalman.observation_matrices     = [1, 0]
	kalman.observation_covariance   = si_r**2
	
	a_hat, d_hat = kalman.em(y).transition_matrices[0, :]

	return a_hat, d_hat


def kalman_logp(y, a_array, d_array, gm):

	# Linear Gaussian dynamic parameters
	mu_a, si_a = gm.mu_a, gm.si_a
	mu_d, si_d = 0, gm.si_d
	si_q, si_r = gm.si_q, gm.si_r

	# Initialise the states according to E[steady state]
	# E[x]_{a~N(mu_a, si_a),d~N(mu_d, si_d)} = E[d] / (1 - E[a]) = 0 
	ss_x  = mu_d / (1 - mu_a)
	# Assuming si_a << 1 we can approximate the covariance by:
	si2_x = si_q**2 / (1 - mu_a**2) + si_d**2 
	
	# 1. Log probability of timeseries y | real and randomly sampled contexts
	# Kalman filter parameters
	kalman = pykalman.KalmanFilter()
	kalman.transition_covariance    = [[si_q**2, 0], [0, 0]]
	kalman.observation_matrices     = [1, 0]
	kalman.observation_covariance   = si_r**2
	
	loglikelihood = np.zeros(len(a_array))
	for n in range(len(loglikelihood)):
		kalman.initial_state_mean       = [0, 1]
		kalman.initial_state_covariance = [[si2_x, 0], [0, 0]]
		kalman.transition_matrices      = [[a_array[n], d_array[n]], [0, 1]]
		loglikelihood[n] = kalman.loglikelihood(y) / len(y)

	# Normalise across the n_ctx contexts
	logp = loglikelihood - scipy.special.logsumexp(loglikelihood)

	return logp


def ss_and_ac_summary(y, a_array, d_array, gm):

	# 1. Steady state distribution for real and randomly sampled contexts
	mu_ss = [d / (1 - a) for a, d in zip(a_array, d_array)]
	si_ss = [(gm.si_r**2 + gm.si_q**2 / (1 - a**2))**0.5 for a in a_array]
	
	# 2. One-step-back autocorrelation for real and randomly sampled contexts
	# Estimate E[] and Var[] of ac = Cov[x_t, x_t-1] / Var[x] for each context
	mu_ac, si_ac = np.zeros(len(a_array)), np.zeros(len(d_array))
	for c in range(len(a_array)):
		x_c, y_c, ac_c = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y)-1)
		x_c[0] = gm._sample_N_(mu_ss[c], si_ss[c])[0]
		y_c[0] = x_c[0] + gm._sample_N_(0, gm.si_r)[0]
		for t in range(len(y)):
			x_c[t]  = a_array[c] * x_c[t-1] + d_array[c] + gm._sample_N_(0, gm.si_q)[0]
			y_c[t]  = x_c[t] + gm._sample_N_(0, gm.si_r)[0]
			ac_c[t-1] = (y_c[t]-mu_ss[c]) * (y_c[t-1]-mu_ss[c]) / si_ss[c]**2
		mu_ac[c], si_ac[c] = ac_c.mean(), ac_c.std()

	return mu_ss, si_ss, mu_ac, si_ac


def ss_and_ac_distributions(y, a_array, d_array, gm):
	
	mu_ss = [d / (1 - a) for a, d in zip(a_array, d_array)]
	si_ss = [(gm.si_r**2 + gm.si_q**2 / (1 - a**2))**0.5 for a in a_array]
	
	ac, ss = np.zeros((len(y), len(a_array))), np.zeros((len(y), len(a_array)))
	for c in range(len(a_array)):
		x_c, y_c, ac_c = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y)-1)
		x_c[0] = gm._sample_N_(mu_ss[c], si_ss[c])[0]
		y_c[0] = x_c[0] + gm._sample_N_(0, gm.si_r)[0]
		for t in range(len(y)):
			x_c[t]  = a_array[c] * x_c[t-1] + d_array[c] + gm._sample_N_(0, gm.si_q)[0]
			y_c[t]  = x_c[t] + gm._sample_N_(0, gm.si_r)[0]
			ss[t-1, c] = y_c[t]
			#ac[t-1, c] = (y_c[t]-mu_ss[c]) * (y_c[t-1]-mu_ss[c]) / si_ss[c]
			ac[t-1, c] = y_c[t] * y_c[t-1]

	return ss, ac


def sample_and_estimate(argin):

	parset, seed, n_trials = argin
	np.random.seed(seed)

	# Generate data and estimate a and d using Kalman EM
	gm = coin.GenerativeModel(parset)
	states, all_a, all_d = gm.sample_states([0] * n_trials, return_pars=True)
	y, a, d = states[0], all_a[0], all_d[0]
	a_hat, d_hat = kalman_estimate_parameters(y, gm)

	# Sample additional contexts 
	n_ctx = 5
	# Sample n_ctx-1 contexts and add (a, d) as parameters for context 0
	a_array = [a] + list(gm._sample_TN_(0, 1, gm.mu_a, gm.si_a, n_ctx-1))
	d_array = [d] + list(gm._sample_N_(0, gm.si_d, n_ctx-1))

	# Measure lopg of the data for the actual and additional contexts:
	logp = kalman_logp(y, a_array, d_array, gm)

	# Distribution of observation stats for the actual and additional contexts:
	mu_ss, si_ss, mu_ac, si_ac = ss_and_ac_summary(y, a_array, d_array, gm)
	ss, ac = ss_and_ac_distributions(y, a_array, d_array, gm)

	return a, d, a_hat, d_hat, logp, ss, ac, mu_ss, si_ss, mu_ac, si_ac


def coin_performance(parset, n_samples, n_trials):
	
	gm = coin.CRFGenerativeModel(parset)
	y, _, c = gm.generate_batch(n_trials, n_samples)

	z, logp, cump, lamb = gm.estimate_coin(y)
	coin_mse = ((z - y[..., 0])**2).mean(1)
	coin_logp = np.log(np.array([np.mean([lamb[b, c[b, t], t] for t in range(n_trials)]) 
							for b in range(n_samples)]))

	base_mse  = (y**2).mean()
	beta = np.array([1/(1+gm.gamma_t) * (gm.gamma_t / (1+gm.gamma_t))**n for n in range(11)])
	base_lopg = np.log(np.array([np.mean([beta[c[b, t]] for t in range(n_trials)]) 
							for b in range(n_samples)]))

	return coin_mse, coin_logp, base_mse, base_lopg


def sample_and_estimate_parset(parset, n_samples, n_trials, pool):

	seeds = np.random.randint(low=1, high=2**14, size=(n_samples)).cumsum()
	results = pool.map(sample_and_estimate, [(parset, s, n_trials) for s in seeds])

	a, d = [r[0] for r in results], [r[1] for r in results]
	a_hat, d_hat = [r[2] for r in results], [r[3] for r in results]
	kal_logp = np.array([r[4] for r in results])
	ss, ac, mu_ss, si_ss, mu_ac, si_ac = [results[0][i] for i in range(5, 11)]

	coin_mse, coin_logp, base_mse, base_logp = coin_performance(parset, n_samples, n_trials)

	parset_stats = {'a': a, 'd': d, 'a_hat': a_hat, 'd_hat': d_hat, 
					'ss': ss, 'ac': ac,
			        'mu_ss': mu_ss, 'si_ss': si_ss, 'mu_ac': mu_ac, 'si_ac': si_ac,
			        'kal_logp': kal_logp, 
			        'coin_logp': coin_logp, 'coin_mse': coin_mse,
			        'base_logp': base_logp, 'base_mse': base_mse}

	return parset_stats


def plot_parset(parset_stats, parset, ax, first=False):
	
	# 1. Kalman perfomance
	a, a_hat, d, d_hat = [parset_stats[k] for k in ['a', 'a_hat', 'd', 'd_hat']]
	alim_up  = 1.05 * max(a + a_hat)
	alim_low = 0.95 * min(a + a_hat)
	dlim_up  = 1.05 * max([abs(v) for v in d + d_hat])

	# 1.1 Estimating a
	ax[0].plot(a, a_hat, 'o', mfc='none')
	ax[0].set_xlabel('ground truth')
	ax[0].set_ylabel(('retention\n' if first else '') + 'estimated')
	ax[0].set_xlim([alim_low, alim_up])
	ax[0].set_ylim([alim_low, alim_up])
	ax[0].plot([0, 1], [0, 1], 'k', transform=ax[0].transAxes)
	ax[0].set_box_aspect(1)
	ax[0].set_title(parset)
	
	# 1.2 Estimating d
	ax[1].plot(d, d_hat, 'o', mfc='none')
	ax[1].set_xlabel('ground truth')
	ax[1].set_ylabel(('drift\n' if first else '') + 'estimated')
	ax[1].set_xlim([-dlim_up, dlim_up])
	ax[1].set_ylim([-dlim_up, dlim_up])
	ax[1].plot([0, 1], [0, 1], 'k', transform=ax[1].transAxes)
	ax[1].set_box_aspect(1)

	# 2. Coin performance
	coin_logp, base_logp = [parset_stats[k] for k in ['coin_logp', 'base_logp']]	
	logp_lim = 1.05 * min(list(base_logp) + list(coin_logp))
	ax[2].plot(base_logp, coin_logp, 'o', mfc='none')
	ax[2].set_xlim([0, logp_lim])
	ax[2].set_ylim([0, logp_lim])
	ax[2].set_xlabel('log p(c|y) E[beta]')
	ax[2].set_ylabel('log p(c|y) COIN')
	ax[2].set_box_aspect(1)
	ax[2].plot([0, 1], [0, 1], '-k', transform=ax[2].transAxes)

	p_rat = np.exp(coin_logp - base_logp)
	ax[3].boxplot(p_rat, vert=False, sym='', notch=True, bootstrap=10000)
	ax[3].axvline(1, color='k')
	ax[3].set_xlabel('p(c|y) COIN / p(c|y) E[beta])')
	ax[3].set_yticks([])
	ax[3].set_box_aspect(1)

	# 3. Steady state and autocorrelation dists for several sampled contexts 
	ss, ac = [parset_stats[k] for k in ['ss', 'ac']]
	ax[4].boxplot(ss, sym='', notch=True, bootstrap=10000)
	ax[4].set_xticks(ticks=[1,2,3,4,5], labels=['true','samp1','samp2','samp3','samp4'])
	ax[4].set_xlabel('context')
	ax[4].set_ylabel('observation value \n (one example sequence)')
	ax[4].set_yticks([])
	ax[4].set_box_aspect(1)

	ax[5].boxplot(ac, sym='', notch=True, bootstrap=10000)
	ax[5].set_xticks(ticks=[1,2,3,4,5], labels=['true','samp1','samp2','samp3','samp4'])
	ax[5].set_xlabel('context')
	ax[5].set_ylabel('autocorrelation \n (one example sequence)')
	ax[5].set_box_aspect(1)

	# 4. Evidences for the ground-truth and other sampled parametrisations
	kal_log = parset_stats['kal_logp']	
	ax[6].boxplot(np.exp(kal_log), sym='', notch=True, bootstrap=10000)
	ax[6].set_xticks(ticks=[1,2,3,4,5], labels=['true','samp1','samp2','samp3','samp4'])
	ax[6].set_xlabel('context')
	ax[6].set_ylabel('p(y|a,b) / sum_[a,b] p(y|a,b)')
	ax[6].set_box_aspect(1)


n_samples = 64
n_trials  = 500
pool = Pool(12)

parsets  = ['validation'] + [f'S{n}' for n in range(1, 9)]
fig, axs = plt.subplots(7, len(parsets))
for n, parset in enumerate(parsets):
	print(f'Estimating {n_samples} for parset "{parset}"" ...', end='')
	t0 = time.time()
	parset_stats = sample_and_estimate_parset(parset, n_samples, n_trials, pool)
	plot_parset(parset_stats, parset, axs[:, n], n==0)
	print(f' done! T={(time.time()-t0):.1f}s')

fig.subplots_adjust(left=0.02,right=0.99,bottom=0.05,top=0.95,wspace=0.1,hspace=0.5)
fig.set_size_inches(30, 18)
plt.savefig('coinpars_separability_S.png')


parsets  = ['validation'] + [f'E{n}' for n in range(1, 9)]
fig, axs = plt.subplots(7, len(parsets))
for n, parset in enumerate(parsets):
	print(f'Estimating {n_samples} for parset "{parset}"" ...', end='')
	t0 = time.time()
	parset_stats = sample_and_estimate_parset(parset, n_samples, n_trials, pool)
	plot_parset(parset_stats, parset, axs[:, n], n==0)
	print(f' done! T={(time.time()-t0):.1f}s')

fig.subplots_adjust(left=0.02,right=0.99,bottom=0.05,top=0.95,wspace=0.1,hspace=0.5)
fig.set_size_inches(30, 18)
plt.savefig('coinpars_separability_E.png')

pool.close()

