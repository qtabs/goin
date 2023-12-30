import random
import numpy as np
import scipy.stats as ss
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.colors
import glob
import os.path
import csv
import matlab.engine
import scipy.optimize
import seaborn as sns
import copy
import pickle
import time
from pathos.multiprocessing import ProcessingPool
import collections.abc
from icecream import ic
from mycolorpy import colorlist as mcp



class GenerativeModel():
    """
    Base class for the Explicit and CRF Generative Model classes. Provides for basic methods
    but it is useless on itself.


    Attributes
    ----------
    parset : str or dict
        as str it identfies a hyperparametrisation compatible with load_pars()
        as dict it should have fields:
        	parset['name']: name to identify the hyperpametrisation
        	parset['pars']: dict with each of the hyperparameters of the generative model

    Methods
    -------
    export_pars()
        Returns a dictionary with the current hyperparametrisation

    generate_batch(n_trials, batch_size)
    	Generates a batch of data; each batch corresponds to a different realisation of the COIN
    	generative model; i.e., same hyperparameters but different parameters

    generate_context_batch(n_trials, batch_size)
    	Generates a batch of context sequences; when used with the explicit implementation also
    	returns the ground truth transition probability matrix and global distribution of contexts

    generate_session(seed, n_trials)
     	Generates a single data sequence, optionally with a user-specified random seed

	sample_observations(contexts, states)
		Generates a single data sequence y_t given a sequence of contexts c_t a sequence of 
		states x_t^c

    sample_states(contexts)
    	Generates state dynamics x_t^c given a sequence of contexts

	estimate_coin(y)
		Runs the COIN inference algorithm and returns the predictions for the observations hat{y}_t,
		the log-probabilities of the actual input sequence y_t, the cumulative probabilities of y_t,
		and the responsibilities of the contexts lambda^c_t
		Requires MATLAB and the COIN inference implementation (https://github.com/jamesheald/COIN)

	run_experiment_coin(experiment, N)
		Generates N batches of fields corresponding to one of the experiments of the COIN and uses
		the COIN inference algorithm to predict the fields; useful for benchmarking

	all_coin_experiments(N)
		Simulates N runs of the COIN inference over all the experiments and returns a dictionary 
		with the predictions and the original fields

	summary_experiments(saveFig, N)
		Runs all_coin_experiments with N runs and plots the results in saveFib

	estimate_leaky_average(y, tau)
		Runs a leaky integrator with integration time constant tau to generate predictions for the
		observations hat{y}_t on the input sequence y_t. Returns the predictions for the 
		observations hat{y}_t, the log-probabilities of the actual input sequence y_t, and the 
		cumulative probabilities of y_t, and tau. If tau is not specified it estimates the best
		value for the current hyperparametrisation.

	fit_best_tau(n_trials)
		Finds the integration time constant tau minimising prediction error for the current
		hyperparametrisation and number of trials.

	plot_example_instances(n_trials, n_instances)
		Plots n_instances random samples with n_trials timepoints from the generative model.

	benchmark(n_trials)
		Performs a thorough benchmarking of the generative model for the given number of trials. 
		It returns a dictionary indicting the performance of the COIN generative model and an
		optimal leaky integrator, and the context and observation sequences used to generate
		the benchmarks. The function stores the benchmarks in a pickle for easy retrieval and only
		performs the computations if the pickle file does not exist.

	measure_ctx_transition_stats(n_trials, )
		Measures the empirical number of visited contexts, context transition probability matrix, 
		and context global probabilities for sequences with n_trials timepoints and the current 
		hyperparametrisation.

    """

	def __init__(self, parset):

		self.pool = ProcessingPool(32)

		if type(parset) is str:
			self.pars   = load_pars(parset)
			self.parset = parset
		else:
			self.pars   = parset['pars']
			self.parset = parset['name']

		self.si_q    = self.pars['si_q']
		self.si_r    = self.pars['si_r']
		self.mu_a    = self.pars['mu_a']
		self.si_a    = self.pars['si_a']
		self.si_d    = self.pars['si_d']
		self.gamma_t = self.pars['gamma_t']
		self.gamma_q = self.pars['gamma_q']
		self.alpha_t = self.pars['alpha_t']
		self.alpha_q = self.pars['alpha_q']
		self.rho_t   = self.pars['rho_t']

	def export_pars(self):
	    """Returns a dictionary with the current hyperparametrisation

	    Returns
	    -------
	    dict
	       dict with each of the hyperparameters of the generative model
	    """

		F = ['si_q','si_r','mu_a','si_a','si_d','gamma_t','gamma_q','rho_t']
		pars = dict([(field, getattr(self, field)) for field in F])

		return pars

	# Sample generation parameters
	def _sample_N_(self, mu, si, N=1):
	    """Samples from a normal distribution

	    Parameters
	    ----------
	    mu : float
	        Mean of the normal distribution
	    si : float
	        Standard deviation of the normal distribution
	    N  : int (optional)
	    	Number of samples

	    Returns
	    -------
	    np.array
	       samples
	    """

		return np.array(ss.norm.rvs(mu, si, N))

	def _sample_TN_(self, a, b, mu, si, N):
	    """Samples from a truncated normal distribution

	    Parameters
	    ----------
	    a  : float
	    	low truncation point
	    b  : float
	    	high truncation point
	    mu : float
	        Mean of the normal distribution before truncation (i.e, location)
	    si : float
	        Standard deviation of the normal distribution before truncation (i.e, size)
	    N  : int (optional)
	    	Number of samples

	    Returns
	    -------
	    np.array
	       samples
	    """

		return np.array(ss.truncnorm.rvs((a-mu)/si, (b-mu)/si, mu, si, N))

	# Sample data
	def generate_batch(self, n_trials, batch_size=1):
	    """Generates a batch of data sampled from the generative model

	    Parameters
	    ----------
	    n_trials   : int
	    	number of time points (i.e., trials in the COIN jargon)
	    batch_size : int
	    	number of batches

	    Returns
	    -------
	    y  : np.array
	    	sequence of observations; dim 0 runs across batches, dim 1 across time points, dim 2 
	    	is set to one for compatibility with pytorch
	    q  : np.array
	    	sequence of cues; same dimensional arrangement as y. Cue emissions are untested.
	    c  : np.array 
	    	sequence of sampled contexts; same dimensional arrangement as y. 
	    """

		self.n_trials = n_trials # hack to avoid passing multiple parameters to pool.map

		# Next line ensures all instances are sampled with different seeds
		seeds = np.random.randint(low=1, high=1024*16, size=(batch_size)).cumsum()
		res = self.pool.map(self.generate_session, seeds)

		y = np.concatenate([r[0][None, ...] for r in res], axis=0)
		q = np.concatenate([r[1][None, ...] for r in res], axis=0)
		c = np.concatenate([r[2][None, ...] for r in res], axis=0)

		self.n_trials = None

		return(y, q, c)

	def generate_context_batch(self, n_trials, batch_size=1):
	    """Generates a batch of context sequences;

	    Parameters
	    ----------
	    n_trials   : int
	    	number of time points (i.e., trials in the COIN jargon)
	    batch_size : int
	    	number of batches

	    Returns
	    -------
	    dict
	    	Field 'C' contains an np.array with the sequences of sampled contexts; same dimensional
	    	arrangement as y in generate_batch
			When used with the explicit implementation also returns:
			Field 'pi': the ground truth transition probability matrix
			Field 'beta': the ground truth global distribution of contexts
	    """

	    # See generate_batch for an explanation of the next two lines
		self.n_trials = n_trials
		seeds = np.random.randint(low=1, high=1024*16, size=(batch_size)).cumsum()

		res = self.pool.map(self.sample_contexts, seeds)

		batch = dict()
		if type(res[0]) is list:
			batch['C'] = np.array([r for r in res])
		elif type(res[0]) is tuple:
			batch['C'] = np.array([r[0] for r in res])
			batch['beta'] = [r[1] for r in res]
			batch['pi'] = [r[2] for r in res]

		self.n_trials = None

		return batch

	def generate_session(self, seed=None, n_trials=None):
	    """Generates a single data sequence, optionally with a user-specified random seed;

	    Parameters
	    ----------
	    seed : int (optional)
	    	random seed for the generation of the sequence (useful for parallelisation)
	    n_trials : int
	    	number of time points (i.e., trials in the COIN jargon)

	    Returns
	    -------
	    y  : np.array
	    	sequence of observations; dim 0 runs across time points, dim 1 is set to one for
	    	compatibility with pytorch
	    q  : np.array
	    	sequence of cues; same dimensional arrangement as y. Cue emissions are untested.
	    c  : np.array 
	    	sequence of sampled contexts; same dimensional arrangement as y. 
	    """

		if n_trials is not None:
			self.n_trials = n_trials

		if seed is not None:
			np.random.seed(seed)

		c = np.zeros((self.n_trials, 1), int)
		q = np.zeros((self.n_trials, 1), int)
		y = np.zeros((self.n_trials, 1))

		contex_sample_results = self.sample_contexts()
		if type(contex_sample_results) is list:
			contexts = contex_sample_results
		elif type(contex_sample_results) is tuple:
			contexts = contex_sample_results[0]
		
		states   = self.sample_states(contexts)
		y[:, 0]  = self.sample_observations(contexts, states)
		q[:, 0]  = self.sample_cues(contexts)
		c[:, 0]  = copy.deepcopy(contexts)

		return(y, q, c)

	def sample_observations(self, contexts, states):
	    """Generates a single data sequence y_t given a sequence of contexts c_t a sequence of 
		states x_t^c

	    Parameters
	    ----------
	    contexts : integer np.array
	    	one-dimensional sequence of contexts 
	    states : dict
	    	dictionary encoding the latent state values (one-dimensional np.array) for each 
	    	context c (keys).

	    Returns
	    -------
	    y  : np.array
	    	one-dimensional sequence of observations
	    """

		y = np.zeros(len(contexts))
		v = self._sample_N_(0, self.si_r, len(contexts))

		for t, c in enumerate(contexts):
			y[t] = states[c][t] + v[t]

		return y

	def sample_states(self, contexts):
	    """Generates a single data sequence y_t given a sequence of contexts c_t a sequence of 
		states x_t^c

	    Parameters
	    ----------
	    contexts : integer np.array
	    	one-dimensional sequence of contexts 

	    Returns
	    -------
	    states : dict
	    	dictionary encoding the latent state values (one-dimensional np.array) for each 
	    	context c (keys).
	    """

	    # Note that retention and drift are sampled in every call
		a = self._sample_TN_(0, 1, self.mu_a, self.si_a, np.max(contexts)+1)
		d = self._sample_N_(0, self.si_d, np.max(contexts)+1)

		states = dict([(c, np.nan * np.ones(len(contexts))) for c in set(contexts)])
		for c in set(contexts):
			i0 = contexts.index(c)
			states[c][i0] = self._sample_N_(d[c]/(1-a[c]), self.si_q/((1-a[c]**2)**.5))
			w = self._sample_N_(0, self.si_q, len(contexts)-i0-1)
			for t in range(i0+1, len(contexts)):
				states[c][t] = a[c] * states[c][t-1] + d[c] + w[t-i0-1]

		return states

	# Coin estimation
	def estimate_coin(self, y, eng=None):
	    """Runs the COIN inference algorithm on a batch of observations
		Requires MATLAB and the COIN inference implementation (https://github.com/jamesheald/COIN)

	    Parameters
	    ----------
	    y   : np.array
	    	sequence of observations; dim 0 runs across batches, dim 1 across time points, dim 2 is 
	    	set to one
		eng : (optional) an instance of matlab.engine.start_matlab() where the COIN paths have been
			added. Useful when estimating multiple problems to avoid running multiple 
			initialisations.

	    Returns
	    -------
	    z_coin : np.array
	    	predictions hat{y}_t for the observations y_{1:t-1}; same dimensional arrangement as y
	    logp   : np.array
			log-probabilities of the input sequence y_t under the COIN posterior distribution; same
			dimensional arrangement as y
	    cump   : np.array
			cumulative probabilities of the input sequence y_t under the COIN posterior 
			distribution; same dimensional arrangement as y. Useful to measure calibration.
	    lamb   : np.array
			responsibilities lambda^c_t for each context c and time-step t. dim 0 runs across 
			batches, dim 1 across time points, dim 2 across contexts (dimension equals the maximum
			number of contexts of the COIN model, currently set to 10+1)

	    """

		if eng is None:
			eng  = matlab.engine.start_matlab()
			eng.addCoinPaths(nargout=0)

		# Translation to the naming of the hyperparameters in James' implementation of the COIN
		# inference algorithm 
		pmCoinDict = {'si_q'   : 'sigma_process_noise',
					  'si_r'   : 'sigma_sensory_noise',
					  'mu_a'   : 'prior_mean_retention',
					  'si_a'   : 'prior_precision_retention',
					  'si_d'   : 'prior_precision_drift',
					  'gamma_t': 'gamma_context',
					  'alpha_t': 'alpha_context', 
					  'rho_t'  : 'rho_context',
					  'gamma_q': 'gamma_cue',
					  'alpha_q': 'alpha_cue'}

		parlist, parvals = [], []
		for p in self.pars:
			if p in pmCoinDict.keys():
				parlist.append(pmCoinDict[p])
				if p in ['si_a', 'si_d']: # James encoded si_a and si_d as precisions
					parvals.append(1/(self.pars[p]**2))
				else:
					parvals.append(self.pars[p])

		Z = eng.runCOIN(matlab.double(y), parlist, parvals, nargout=4)
		
		z_coin, logp, cump, lamb = np.array(Z[0]), np.array(Z[1]), np.array(Z[2]), np.array(Z[3])

		return(z_coin, logp, cump, lamb)

	def run_experiment_coin(self, experiment, N=20, eng=None):
	    """Runs the COIN inference algorithm on the field sequence corresponding to one of the 
	    experiments of original COIN paper.

	    Parameters
	    ----------
	    experiment : str
	    	Either 'evoked' (recovery), 'spontaneous' (recovery), 'savings', (retrograde) 
	    	'interference', or (effect of environmental) 'consistency' (on learning rate)
	    N   : int (optional)
	    	Number of runs
		eng : (optional) an instance of matlab.engine.start_matlab() where the COIN paths have been
			added. Useful when estimating multiple problems to avoid running multiple 
			initialisations.
		ax  : (optional) an instance of plt.subplots()
			plt axes where to plot the result of the simulation
	    
	    Returns
	    -------
	    u  : dict() (optional)
	    	A dictionary encoding the predictions across the experiment conditions (keys); same 
	    	structure as each of the sub-dicts of U produced by initialise_experiments_U
	    x0  : np.array
			field values for one of the runs; dim 0 is set to 1, dim 1 runs across time points, 
			dim 2 is set to one. Useful for plotting.
	    """

		if eng is None:
			eng = matlab.engine.start_matlab()
			eng.addCoinPaths(nargout=0)

		match experiment:

			case 'consistency':
				x  = generate_field_sequence(experiment, self.si_r, 1, pStay=0.5)
				nanix = np.where(np.isnan(x[0]))[0]
				triplets = [ix for ix in nanix if ix+2 in nanix]
				x0 = x[0, [t+1 for t in triplets], :]

				pStayList, u = [0.1, 0.5, 0.9], dict()
				for p in pStayList:
					x  = generate_field_sequence(experiment, self.si_r, N, pStay=p)
					u0 = self.estimate_coin(x, eng)[0]
					u[p] = np.stack([u0[:, t+2]-u0[:,t] for t in triplets], axis=1)

			case 'interference':
				nPlusList, u = [0, 13, 41, 112, 230, 369], dict()
				for nPlus in nPlusList:
					x = generate_field_sequence(experiment, self.si_r, N, Np=nPlus)
					u[nPlus] = self.estimate_coin(x, eng)[0][:, (160+nPlus):]
				x0 = x[0, 160+nPlus:, 0]

			case 'savings':
				x = generate_field_sequence(experiment, self.si_r, N)
				u0 = self.estimate_coin(x, eng)[0]
				t0, t1, dur = 60, 60+125+15+50+50+60, 125
				u = {'first': u0[:, t0:(t0+dur)], 'second': u0[:, t1:(t1+dur)]}
				x0 = x[0, t0:(t0+dur), 0]

			case 'spontaneous' | 'evoked':
				x = generate_field_sequence(experiment, self.si_r, N)
				u = self.estimate_coin(x, eng)[0]
				x0 = x[0, :, 0]

		if axs is not None:
			plot_experiment(u, experiment, axs=axs, cmap="Greens")

		return(u, x0)

	def all_coin_experiments(self, N=20, eng=None):
	    """Runs the COIN inference algorithm on the field sequence corresponding to the five 
	    experiments of the original COIN paper that do not involve cue emissions. Results are 
	    stored in a pickle path and retrieved in each call if the file exists.

	    Parameters
	    ----------
	    N   : int (optional)
	    	Number of runs
		eng : (optional) an instance of matlab.engine.start_matlab() where the COIN paths have been
			added. Useful when estimating multiple problems to avoid running multiple 
			initialisations.
	    
	    Returns
	    -------
	    U   : dict
			Dictionary with np.arrays encoding the predictions hat{y}_t for the fields of each
			experiment (keys); dim 0 runs across runs, dim 1 across time points, dim 2 set to one.
	    x0  : dict
	    	Dictionary with np.arrays encoding the field values for one of the runs for each 
	    	experiment (keys); dim 0 is set to 1, dim 1 runs across time points, dim 2 set to one.
	    """

	    # Pickle storage path is hardcoded to: 
		storingpath = f'./benchmarks/{self.parset}-coinexp.pickle'

		if os.path.exists(storingpath):
			with open(storingpath, 'rb') as f:
				experiments_kit = pickle.load(f)
				U, X0 = experiments_kit['U'], experiments_kit['X0']
		else:
			if eng is None:
				eng = matlab.engine.start_matlab()
				eng.addCoinPaths(nargout=0)
				
			experiments = ['spontaneous', 'evoked', 'savings', 'interference', 'consistency']
			U, X0 = dict(), dict()
			for exp in experiments:
				U[exp], X0[exp] = self.run_experiment_coin(exp, N, eng)
			experiments_kit = {'U': U, 'X0': X0}

			with open(storingpath, 'wb') as f:
				pickle.dump(experiments_kit, f)

		return U, X0

	def summary_experiments(self, savefig=None, N=20, cmap="Greens", axs=None, eng=None):
	    """Plots the predictions of the COIN inference algorithm for the field sequences 
	    corresponding to the five experiments of the original COIN paper that do not involve cue 
	    emissions.

	    Parameters
	    ----------
	    savefig : str (optional)
	    	Path where to store the figure (without extension). If savefig is not provided, 
	    	results are plotted but not printed (not shown; use plt.show() for that).
	    N    : int (optional)
	    	Number of runs.
		cmap : (optional) string
			mycolorpy.colorlistp colormap to use for the plots; e.g., 'Blues', 'Greens', etc
		axs  : (optional) an np.array instance of plt.subplots()
			plt axes where to plot the results
		eng  : (optional) an instance of matlab.engine.start_matlab() where the COIN paths have been
			added. Useful when estimating multiple problems to avoid running multiple 
			initialisations.
	    """

		U, X0 = self.all_coin_experiments(N, eng)

		if axs is None:
			fig, axs = plt.subplots(1, 5)
		else:
			fig = None

		for experiment, ax in zip(U, axs):
			plot_experiment(U[experiment], experiment, ax, cmap)

		if savefig is not None and fig is not None:
			fig.subplots_adjust(left=0.05,right=0.98,bottom=0.17,top=0.9,wspace=0.3,hspace=0.3)
			fig.set_size_inches(20, 3)
			plt.savefig(f'{savefig}.png')

	# Leaky integrator estimation
	def estimate_leaky_average(self, y, tau=None):
	    """Runs a leaky integrator with integration time constant tau to generate predictions for
	    the observations hat{y}_t on the input sequence y_{1:t-1}.

	    Parameters
	    ----------
	    y   : np.array
	    	sequence of observations; dim 0 runs across batches, dim 1 across time points, dim 2 is 
	    	set to one 
		tau : (optional) float or np.array
			(set of) integration time constant(s). If not specified, it's set to the optimal value
			for the current hyperparametrisation.

	    Returns
	    -------
	    z_slid : np.array
	    	predictions hat{y}_t for the observations y_{1:t-1}; same dimensional arrangement as y
	    logp   : np.array
			log-probabilities of the input sequence y_t
	    cump   : np.array
			cumulative probabilities of the input sequence y_t. Useful to measure calibration.
	    tau   : np.array
			integration time constant(s); useful if the optimal value is estimated during the call.

	    """
		if tau is None:
			tau = self.fit_best_tau(n_trials = y.shape[1])

		if type(tau) != np.ndarray:
			tau = np.array([tau])

		weights = np.zeros((y.shape[1], len(tau)))
		weights[:, tau>0] = np.exp(- np.einsum('T,m->Tm', np.arange(y.shape[1], 0, -1), 1/tau[tau>0]))
		for tix in np.where(tau<=0)[0]:
			weights[:,tix] = np.eye(y.shape[1])[-1]
		
		z_slid, s_slid = np.zeros((y.shape[0],) + weights.shape), np.ones((y.shape[0],) + weights.shape)

		for t in range(1, y.shape[1]):
			w = weights[-t:,:] / weights[-t:, :].sum(0)
			z_slid[:, t, :] = np.einsum('bT,Tm->bm', y[:,:t], w)
			s_slid[:, t, :] = np.sqrt(np.einsum('bTm,Tm->bm', (y[:, :t, np.newaxis] - z_slid[:, :t,:])**2, w))

		logp = (-0.5 * np.log(2*np.pi) -np.log(s_slid) - 0.5 * ((z_slid - y[:,:,np.newaxis]) / s_slid) ** 2)
		cump = 0.5 * (1 + scipy.special.erf((y[:,:,np.newaxis] - z_slid) / (np.sqrt(2) * s_slid)))

		return(z_slid, logp, cump, tau)

	def _estimate_leaky_average_call_(self, inpars):
		""" Wrapper of estimate_leaky_average to take in multiple parameters during 
		parallelisation; not meant as a user-end method"""

		x, tau = inpars
		return(self.estimate_leaky_average(x, tau=tau))

	def fit_best_tau(self, n_trials=5000, n_train_instances=500):
		"""Finds the integration time constant tau minimising prediction error for the current
		hyperparametrisation and number of trials.

	    Parameters
	    ----------
	    n_trials : int
	    	number of time points (i.e., trials in the COIN jargon)
		n_train_instances : (optional) int
	    	number of instances to use for the estimation

	    Returns
	    -------
	    best_t : float
	    	optimal integration time constant tau
	    """

		def fn(tau):
			res = self.pool.map(self._estimate_leaky_average_call_, [(x[None, ...], tau) for x in X])
			mse = np.mean([((r[0][:,:,0] - x)**2).mean() for r, x in zip(res, X)])
			return(mse)

		X = self.generate_batch(n_trials, n_train_instances)[0][:, :, 0]

		optimRes = scipy.optimize.minimize_scalar(fn, bounds=(0, X.shape[1]), 
													 method='bounded', 
													 options={'xatol':1e-3})				
		if optimRes.success:
			best_t = optimRes.x
		else:
			taus = np.arange(100, -1, -1)
			res  = self.pool.map(self._estimate_leaky_average_call_, [(x, taus) for x in X])
			z_slid = np.array([r[0] for r in res])
			best_t = taus[((z_slid[:, 0, :, :] - X[..., None])**2).mean((0,1)).argmin()]

		return best_t

	# Example plotting
	def plot_example_instances(self, n_trials=2000, n_instances=16, suffix=None):
		"""Plots n_instances random samples with n_trials timepoints from the generative model 
		and the empirical context transition probability matrix and global probabilities averaged
		across instances. The two plots are saved as figures.

	    Parameters
	    ----------
	    n_trials : int
	    	number of time points (i.e., trials in the COIN jargon)
		n_instances : (optional) int
	    	number of independent instances to plot
	    suffix : (optional) str
	    	suffix appended to the filepath of the two produced figures
	    """

		n = int(np.floor(np.sqrt(n_instances)))
		m = int(np.ceil(n_instances / n))
		
		fig1, axs1 = plt.subplots(n, m)
		y, q, c = self.generate_batch(n_trials, n*m)

		ctx = list(set(c.reshape(-1)))

		time = np.arange(y.shape[1], dtype=int)
		colours = list(matplotlib.colors.TABLEAU_COLORS.values()) * int(np.ceil(len(ctx)/10))
		ylim = np.ceil(10 * np.abs(y).max()) / 10

		for b, ax in enumerate(axs1.reshape(-1)):

			T = [0] + [t for t in range(1, n_trials) if (c[b,t,0]-c[b,t-1,0])!=0] + [n_trials]
			for t0, t1 in zip(T[:-1], T[1:]):
				ax.plot(time[t0:t1], y[b, t0:t1, 0], color = colours[c[b, t0, 0]])
			ax.set_ylim((-ylim, +ylim))			
			ax.set_xlabel('trial number')
			ax.set_ylabel('field / output')


		fig2, axs2 = plt.subplots(n, m)

		for b, ax in enumerate(axs2.reshape(-1)):
			pi=np.array([[np.logical_and(c[b,:-1]==k,c[b,1:]==l).mean() for k in ctx] for l in ctx])
			sns.heatmap(pi, annot=True, ax=ax)

		fig1.set_size_inches(29, 13)
		fig2.set_size_inches(13, 11)
		fig1.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.3,hspace=0.3)
		fig2.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.3,hspace=0.3)
		fig1.savefig(f'samples-ys{"-{suffix}" if suffix is not None else ""}.png')
		fig2.savefig(f'samples-pi{"-{suffix}" if suffix is not None else ""}.png')

	# Benchmarks
	def benchmark(self, n_trials=5000, n_instances=512, suffix=None):
		"""Performs a thorough benchmarking of the generative model for the given number of trials. 
		The function stores the benchmarks in a pickle for easy retrieval and only performs the
		computations if the pickle file does not exist.

	    Parameters
	    ----------
	    n_trials : int
	    	number of time points (i.e., trials in the COIN jargon)
		n_instances : (optional) int
	    	number of instances used to calculate the benchmark
	    suffix : (optional) str
	    	suffix appended to the filepath of the pickle file; useful to keep e.g. independent
	    	training and testing benchmarking sets

	    Returns
	    -------
		benchmark_kit : dict
			A dictionary with fields:
			benchmark_kit['X']      : set of observation sequences used to compute the benchmarks
			benchmark_kit['C']      : associated set of context sequences
			benchmark_kit['perf']   : a dictionary with several statistics measuring the 
									  performance of the COIN and leaky integrator on the set X
			benchmark_kit['best_t'] : optimal integration time constant of the leaky integrator
	    """

		if suffix is None:
			benchmarkpath = f'./benchmarks/{self.parset}-{n_trials}.pickle'
		else:
			benchmarkpath = f'./benchmarks/{self.parset}-{n_trials}-{suffix}.pickle'

		if os.path.exists(benchmarkpath):
			with open(benchmarkpath, 'rb') as f:
				benchmark_kit = pickle.load(f)
		else:
			minloglik = -100

			print(f'### Computing benchmarks for {self.parset} ###')
			t0 = time.time()

			print(f'Finding best tau...', end=' ', flush=True)
			tau = self.fit_best_tau(n_trials)
			print(f'[best_t = {tau:.1f}]', end=' ', flush=True)

			print(f'Generating data...', end=' ', flush=True)
			X, Q, C = self.generate_batch(n_trials, n_instances)

			print(f'Estimating LI...', end=' ', flush=True)
			res = self.pool.map(self._estimate_leaky_average_call_, [(x[None, :, 0], tau) for x in X])

			z_slid    = np.array([r[0][0, :, 0] for r in res])
			p_slid    = np.array([r[1][0, :, 0] for r in res])
			cump_slid = np.array([r[2][0, :, 0] for r in res])

			F = np.linspace(0, 1, 1000)
			cums_cump_slid = np.array([(cump_slid <= f).sum(1)/n_trials for f in F])
			LI_mse = ((z_slid - X[..., 0])**2).mean(1)
			LI_kol = abs(cums_cump_slid - F[:, None]).max(0)
			LI_ce  = p_slid.mean(1)
			LI_ct_ac = (C[:, :, 0] == 0).mean(1) # LI assumed to predict always context 0
			LI_ct_p  = (C[:, :, 0] == 0).mean(1)
			LI_ct_ce = minloglik * (C[:, :, 0] != 0).mean(1)


			print(f'Estimating coin...', flush=True)
			eng  = matlab.engine.start_matlab()
			eng.addCoinPaths(nargout=0)
			z_coin, ll_coin, cump_coin, lamb = self.estimate_coin(X, eng)
			loglamb = np.log(lamb + np.exp(minloglik))

			coin_mse = ((z_coin - X[..., 0])**2).mean(1)
			cums_cump_coin = np.array([(cump_coin <= f).sum(1)/cump_coin.shape[1] for f in F])
			coin_kol = abs(cums_cump_coin - F[:, np.newaxis]).max(0)
			coin_ce  = ll_coin.mean(1)

			c_hat = np.argmax(lamb, axis=1)
			coin_ct_ac = (c_hat == C[..., 0]).mean(1)
			coin_ct_p  = np.zeros(C.shape[0])
			coin_ct_ce = np.zeros(C.shape[0])

			for b in range(C.shape[0]):
				for ctx in range(lamb.shape[1]):
					ctx_ix = np.where(C[b, :, 0] ==ctx)[0]
					coin_ct_p[b] += np.nansum(lamb[b, ctx, ctx_ix]) / C.shape[1]
					coin_ct_ce[b] += np.nansum(loglamb[b, ctx, ctx_ix]) / C.shape[1]


			perf = {'LI' :  {'mse': {}, 'kol': {}, 'ce': {}, 'ct_ac': {}, 'ct_p': {}, 'ct_ce': {}}, 
					'coin': {'mse': {}, 'kol': {}, 'ce': {}, 'ct_ac': {}, 'ct_p': {}, 'ct_ce': {}}}

			# Observations MSE
			perf['LI']['mse']['avg']      = LI_mse.mean()
			perf['LI']['mse']['sem']      = LI_mse.std() / np.sqrt(n_instances)
			# Observations calibration
			perf['LI']['kol']['avg']      = LI_kol.mean()
			perf['LI']['kol']['sem']      = LI_kol.std() / np.sqrt(n_instances)
			# Observations cross-entropy
			perf['LI']['ce']['avg']       = LI_ce.mean()
			perf['LI']['ce']['sem']       = LI_ce.std() / np.sqrt(n_instances)
			# Context identification accuracy
			perf['LI']['ct_ac']['avg']    = LI_ct_ac.mean()
			perf['LI']['ct_ac']['sem']    = LI_ct_ac.std() / np.sqrt(n_instances)
			# Probability of the actual context on the posterior of the prediction
			perf['LI']['ct_p']['avg']     = LI_ct_p.mean()
			perf['LI']['ct_p']['sem']     = LI_ct_p.std() / np.sqrt(n_instances)
			# Context cross-entropy
			perf['LI']['ct_ce']['avg']    = LI_ct_ce.mean()
			perf['LI']['ct_ce']['sem']    = LI_ct_ce.std() / np.sqrt(n_instances)

			perf['coin']['mse']['avg']    = coin_mse.mean()
			perf['coin']['mse']['sem']    = coin_mse.std() / np.sqrt(n_instances)
			perf['coin']['kol']['avg']    = coin_kol.mean()
			perf['coin']['kol']['sem']    = coin_kol.std() / np.sqrt(n_instances)
			perf['coin']['ce']['avg']     = coin_ce.mean()
			perf['coin']['ce']['sem']     = coin_ce.std() / np.sqrt(n_instances)
			perf['coin']['ct_ac']['avg']  = coin_ct_ac.mean()
			perf['coin']['ct_ac']['sem']  = coin_ct_ac.std() / np.sqrt(n_instances)
			perf['coin']['ct_p']['avg']   = coin_ct_p.mean()
			perf['coin']['ct_p']['sem']   = coin_ct_p.std() / np.sqrt(n_instances)
			perf['coin']['ct_ce']['avg']  = coin_ct_ce.mean()
			perf['coin']['ct_ce']['sem']  = coin_ct_ce.std() / np.sqrt(n_instances)

			rat = perf['LI']['mse']['avg'] / perf['coin']['mse']['avg']
			dif = perf['coin']['ct_ce']['avg'] - perf['LI']['ct_ce']['avg']
			print(f'done! Time {(time.time()-t0)/60:.1f}m', end='; ')
			print(f'mse_coin/mse_slid = {rat:.2f}', end='; ')
			print(f'ce_coin - ce_slid = {dif:.2f}')

			benchmark_kit = {'X': X, 'C': C, 'perf': perf, 'best_t': tau}

			with open(benchmarkpath, 'wb') as f:
				pickle.dump(benchmark_kit, f)

			print(f'###################################')
			print()

		return benchmark_kit

	# Measuring and validation plots
	def measure_ctx_transition_stats(self, n_trials=10000, n_instances=1024):
		"""Measures the empirical number of visited contexts, context transition probability matrix, 
		and context global probabilities for sequences with n_trials timepoints and the current 
		hyperparametrisation.

	    Parameters
	    ----------
	    n_trials : int
	    	number of time points (i.e., trials in the COIN jargon)
		n_instances : (optional) int
	    	number of instances used to calculate the stats

	    Returns
	    -------
		r : dict
			A dictionary with fields:
			r['n_sampled']      : a dict encoding the dependency of visited contexts with n_trials
			r['empirical_pi']   : average empirically measured transition probability matrix
			r['empirical_beta'] : average empirically measured global context probabilities
			When used under ExplicitGenerativeModel the dictionary has, in addition:
			r['ground_pi']   : average ground-truth transition probability matrix
			r['ground_beta'] : average ground-truth global context probabilities
	    """

		# Data generation
		batch = self.generate_context_batch(n_trials, n_instances)
		C = batch['C']

		# N ctxs sampled at time t
		ticks = [int(t) for t in np.logspace(0, np.log10(n_trials), int(2*np.log10(n_trials)+1))]
		n_sampled = np.array([C[:, :t].max(1)+1 for t in ticks]).T
		n_sampled_avg, n_sampled_sem = n_sampled.mean(0), n_sampled.std(0) /  np.sqrt(n_instances)

		# Empirical beta
		ctxs, n_ctx = list(set(C.reshape(-1))), C.max()+1
		empirical_beta = [(C==c).mean() for c in ctxs]

		# Empirical pi
		empirical_pi = np.array([[np.logical_and(C[:, :-1]==c1, C[:, 1:]==c2).mean() for c1 in ctxs] 
											                                        for c2 in ctxs])
		r = dict()
		if 'pi' in batch:
			pi_avg = np.zeros((n_ctx, n_ctx))
			for pi in batch['pi']:
				n = min(pi.shape[0], n_ctx)
				pi_avg[:n, :n] += pi[:n, :n] / n_instances
			r['ground_pi'] = pi_avg

		if 'beta' in batch:
			beta_avg = np.zeros(n_ctx)
			for beta in batch['beta']:
				n = min(beta.shape[0], n_ctx)
				beta_avg[:n] += beta[:n] / n_instances
			r['ground_beta'] = beta_avg

		# Storing
		r['n_sampled'] = {'ticks': ticks, 'avg': n_sampled_avg, 'sem': n_sampled_sem}
		r['empirical_pi']   = empirical_pi
		r['empirical_beta'] = empirical_beta

		return r


class ExplicitGenerativeModel(GenerativeModel):
    """
    Extends GenerativeModel to include context and cue sampling using the explicit method: i.e., 
    first sampling the transition probability matrices and then simulating a Markov Chain. This 
    method is an approximation. Cue emission has not been thoroughly tested.

    Additional Methods
    -------
	sample_pi_t(alpha, beta, rho)
	    Samples a finite approximation of the ground truth transition probability matrix

	sample_pi_q(alpha, beta, N):
	    Samples a finite approximation of the ground truth cue emission probability matrix.

	sample_contexts(seed, n_trial):
	    Samples a sequence of contexts by simulating a markov chain. Each call uses an 
	    independent sample of pi_t and beta_t.

	sample_cues(contexts):
	    Samples a sequence of cues. Each call uses an independent sample of pi_q and beta_q.

    """

	def __init__(self, parset):
		super().__init__(parset)

	# Generic sampling methods
	def _sample_DP_(self, alpha, H):
	    """Samples from a dirichlet process

	    Parameters
	    ----------
	    alpha : float
	        concentration parameter
	    H     : np.array
	        base distribution (a discrete probability distribution)

	    Returns
	    -------
	    np.array
	       sample (a discrete probability distribution)
	    """

		pi_tilde = self._sample_GEM_(alpha, threshold=1E-5)
		theta    = random.choices(range(len(H)), H, k=len(pi_tilde))
		G = pi_tilde @ np.eye(len(H))[theta]

		return G

	def _sample_GEM_(self, gamma, threshold=1E-8):
	    """Samples from a GEM distribution using the stick-breaking construction

	    Parameters
	    ----------
	    gamma     : float
	        concentration parameter (controls the relative length of each break point) 
	    threshold : float
	        maximum stick length before stopping the theoretically-infinite process

	    Returns
	    -------
	    np.array
	       sample (a discrete probability distribution)
	    """

		sample, stick_len = [], 1.

		while stick_len > threshold:
			beta_sample = ss.beta.rvs(1, gamma)
			sample.append(stick_len * beta_sample)
			stick_len *= 1 - beta_sample

		 # Samples are renormalised; alternatively we could create a last segment with stick_len
		sample = np.array(sample) / np.sum(sample)

		return sample

	# Parameter sampling methods
	def sample_pi_t(self, alpha, beta, rho):
	    """Samples a finite approximation of the ground truth transition probability matrix

	    Parameters
	    ----------
	    alpha  : float
	        concentration parameter controls the dispersion of the rows with respect to beta 
	    beta : np.array (a discrete probability distribution)
	        global probabilities
		rho : float 
			normalised self-transition bias; 0 < rho < 1
	    
	    Returns
	    -------
	    np.array
	       transition probability matrix; row n is the transition distribution from context n
	    """

		pi = np.zeros((len(beta), len(beta)))
		for j in range(len(beta)):
			delta = np.eye(len(beta))[j]
			pi[j, :] = self._sample_DP_(alpha / (1-rho), (1-rho) * beta + rho * delta)

		return pi

	def sample_pi_q(self, alpha, beta, N):
	    """Samples a finite approximation of the ground truth cue emission probability matrix.

	    Parameters
	    ----------
	    alpha  : float
	        concentration parameter controls the dispersion of the rows with respect to beta 
	    beta : np.array (a discrete probability distribution)
	        global cue probabilities
		N : int 
			number of contexts
	    
	    Returns
	    -------
	    np.array
	       cue emission probability matrix; row n is the probability distribution across cues for
	       context n
	    """

		pi = np.zeros((N, len(beta)))
		for j in range(N):
			pi[j, :] = self._sample_DP_(alpha, beta)

		return pi

	# Observation sampling methods
	def sample_contexts(self, seed=None, n_trials=None):
	    """Samples a sequence of contexts by simulating a markov chain. Each call uses an 
	    independent sample of pi_t and beta_t.

	    Parameters
	    ----------
	    seed : int (optional)
	    	random seed for the generation of the sequence (useful for parallelisation)
	    n_trials : int (optional)
	    	number of time points (i.e., trials in the COIN jargon). If not specified, n_trials 
	    	is readout from self.n_trials

	    Returns
	    -------
	    contexts : np.array 
	    	sequence of sampled contexts; dim 0 runs across time points, dim 1 is set to one for
	    	compatibility with pytorch
 		beta_t   : np.array
 			ground-truth global context probabilities
 		pi_t     : np.array
 			ground-truth transition probability matrix
	    """

		if seed is not None:
			np.random.seed(seed)
		if n_trials is not None: 
			self.n_trials = n_trials

		beta_t = self._sample_GEM_(self.pars['gamma_t'])
		pi_t   = self.sample_pi_t(self.pars['alpha_t'], beta_t, self.pars['rho_t'])

		# First context sampled from the global distribution
		c = random.choices(range(len(beta_t)), beta_t)

		n_ctx = len(beta_t)
		for t in range(1, self.n_trials):
			c += random.choices(range(n_ctx), pi_t[c[t-1], :])

		# Next 12 lines re-order the context indices so that they always appear on ascending 
		# order (ToDo: return also non-reordered list for the computation of average empirical pi)
		c2c = dict()
		contexts = np.zeros(self.n_trials, dtype=int)
		next_c = 0

		for t in range(self.n_trials):
			if c[t] not in c2c.keys():
				c2c[c[t]] = next_c
				next_c += 1
			contexts[t] = c2c[c[t]]

		contexts = list(contexts)

		self.beta_t = beta_t
		self.pi_t   = pi_t
		self.n_ctx  = n_ctx

		return(contexts, beta_t, pi_t)

	def sample_cues(self, contexts):
	    """Samples a sequence of cues. Each call uses an independent sample of pi_q and beta_q.
	    Cue emission has not been thoroughly tested

	    Parameters
	    ----------
	    contexts : np.array
	    	sequence of sampled contexts; dim 0 runs across time points, dim 1 is set to one

	    Returns
	    -------
	    cues : np.array 
	    	sequence of emitted cues
	    """

		beta_q  = self._sample_GEM_(self.pars['gamma_q'])
		pi_q    = self.sample_pi_q(self.pars['alpha_q'], beta_q, max(contexts)+1)
		n_cues  = len(beta_q)

		cues = [random.choices(range(n_cues), pi_q[c, :])[0] for c in contexts]

		self.beta_q = beta_q
		self.pi_q   = pi_q
		self.n_cues  = n_cues

		return cues


class CRFGenerativeModel(GenerativeModel):
    """
    Extends GenerativeModel to include context and cue sampling using the Chinese Restaurant 
    Franchise method. This method is exact. Cue emission has not been thoroughly tested.

    Additional Methods
    -------
	sample_contexts(seed=None, n_trials=None):
	    Samples a sequence of contexts using the CRF construction.

	sample_cues(contexts):
	    Samples a sequence of cues.
	"""

	def __init__(self, parset):
		super().__init__(parset)

	# Observation sampling methods
	def sample_contexts(self, seed=None, n_trials=None):
		"""Samples a sequence of contexts by simulating a markov chain. Each call uses an 
	    independent sample of pi_t and beta_t.

	    Parameters
	    ----------
	    seed : int (optional)
	    	random seed for the generation of the sequence (useful for parallelisation)
	    n_trials : int
	    	number of time points (i.e., trials in the COIN jargon). If not specified, n_trials 
	    	is readout from self.n_trials

	    Returns
	    -------
	    contexts : np.array 
	    	sequence of sampled contexts; dim 0 runs across time points, dim 1 is set to one for
	    	compatibility with pytorch
	    """

		if seed is not None:
			np.random.seed(seed)
		if n_trials is not None: 
			self.n_trials = n_trials

		max_j = 2
		max_t = 2
		c = np.zeros(self.n_trials, dtype=int)

		# M[j, k]  = n of tables serving dish k in restaurant j
		# Mb[j, k] = n of tables having considered dish k in restaurant j
		M, Mb = np.zeros((max_j, max_j), dtype=int), np.zeros((max_j, max_j), dtype=int)
		
		# N[j, l]  = n of customers in restaurant j sitting at table l
		# D[j, l]  = dish served at table l of restaurant j
		N, D = np.zeros((max_j, max_t), dtype=int), -np.ones((max_j, max_t), dtype=int)

		# Very first customer:
		j, table = 0, 0
		N[j, table] += 1 

		weights_dish = np.array([1, self.gamma_t])
		considered_dish = random.choices(range(len(weights_dish)), weights_dish)[0]
		dish = random.choices([considered_dish, c[0]], [1-self.rho_t, self.rho_t])[0]
		D[j, table] = dish
		c[1] = dish

		for t in range(1, self.n_trials-1):
			
			# Restaurant
			j = c[t]

			# Table
			weights_table = np.append(N[j, N[j]>0], self.alpha_t/(1-self.rho_t))
			table = random.choices(range(len(weights_table)), weights_table)[0]
			if table >= N.shape[1]:
				N = np.append(N, np.zeros((N.shape[0], max_t), dtype=int), 1)
				D = np.append(D, -np.ones((D.shape[0], max_t), dtype=int), 1)
			N[j, table] += 1 

			# Dish
			if D[j, table] == -1: # If the table has no dish we sample a new one
				weights_dish = np.append(Mb.sum(0)[Mb.sum(0)>0], self.gamma_t)
				considered_dish = random.choices(range(len(weights_dish)), weights_dish)[0]
				
				if considered_dish >= Mb.shape[1]:
					N  = np.append(N,  np.zeros((max_j, N.shape[1]),  dtype=int), 0)
					D  = np.append(D,  -np.ones((max_j, D.shape[1]),  dtype=int), 0)
					M  = np.append(M,  np.zeros((max_j, M.shape[1]),  dtype=int), 0)
					Mb = np.append(Mb, np.zeros((max_j, Mb.shape[1]), dtype=int), 0)
					M  = np.append(M,  np.zeros((M.shape[0], max_j),  dtype=int), 1)
					Mb = np.append(Mb, np.zeros((Mb.shape[0], max_j), dtype=int), 1)

				dish = random.choices([considered_dish, j], [1-self.rho_t, self.rho_t])[0]
				Mb[j, considered_dish] += 1
				M[j, dish] += 1
				D[j, table] = dish
			else: # Else we take the assigned dish
				dish = D[j, table]

			# Next context is current dish
			c[t+1] = dish

		# Order contexts from 0 to n_contexts
		c2c = dict()
		contexts = np.zeros(self.n_trials, dtype=int)
		n_contexts = 0

		for t in range(self.n_trials):
			if c[t] not in c2c.keys():
				c2c[c[t]] = n_contexts
				n_contexts += 1
			contexts[t] = c2c[c[t]]

		return list(contexts)

	def sample_cues(self, contexts):
	    """Samples a sequence of cues. Each call uses an independent sample of pi_q and beta_q.
	    Cue emission has not been thoroughly tested

	    Parameters
	    ----------
	    contexts : np.array
	    	sequence of sampled contexts; dim 0 runs across time points, dim 1 is set to one

	    Returns
	    -------
	    cues : np.array 
	    	sequence of emitted cues
	    """
	    
		max_j = max(contexts) + 1
		max_d = max(5, max(contexts) + 1)
		max_t = 500

		n_trials = len(contexts)
		q = np.zeros(n_trials, dtype=int)

		# M[j, k]  = n of tables serving dish k in restaurant j
		M = np.zeros((max_j, max_d), dtype=int)
		
		# N[j, l]  = n of customers in restaurant j sitting at table l
		# D[j, l]  = dish served at table l of restaurant j
		N, D = np.zeros((max_j, max_t), dtype=int), -np.ones((max_j, max_t), dtype=int)

		# Very first customer:
		j, table = 0, 0
		N[j, table] += 1 
		weights_dish = np.array([1, self.gamma_q])
		dish = random.choices(range(len(weights_dish)), weights_dish)[0]
		D[j, table] = dish
		q[0] = dish

		for t in range(n_trials):
			
			# Restaurant
			j = contexts[t]

			# Table
			weights_table = np.append(N[j, N[j]>0], self.alpha_q)
			table = random.choices(range(len(weights_table)), weights_table)[0]
			if table >= N.shape[1]:
				N = np.append(N, np.zeros((max_j, max_t), dtype=int), 1)
				D = np.append(D, -np.ones((max_j, max_t), dtype=int), 1)

			N[j, table] += 1 

			# Dish
			if D[j, table] == -1: # If the table has no dish we sample a new one
				weights_dish = np.append(M.sum(0)[M.sum(0)>0], self.gamma_q)
				dish = random.choices(range(len(weights_dish)), weights_dish)[0]
				if dish >= M.shape[1]:
					M = np.append(M, np.zeros((max_j, max_d), dtype=int), 1)
				M[j, dish] += 1
				D[j, table] = dish
			else: # Else we take the assigned dish
				dish = D[j, table]

			# emitted cue is current dish
			q[t] = dish

		# Order cues from 0 to n_cues
		q2q    = dict()
		cues   = np.zeros(n_trials, dtype=int)
		n_cues = 0

		for t in range(n_trials):
			if q[t] not in q2q.keys():
				q2q[q[t]] = n_cues
				n_cues += 1
			cues[t] = q2q[q[t]]

		return cues


def load_pars(parset):
	""" Loads a set of COIN hyperparameters from a pre-specified label. 

	    Parameters
	    ----------
	    parset : str
	    	String identifying the hyperparametrisation. Can be: 1) a reference from a subject from
	    	the COIN paper (e.g., 'S1', 'E5', 'M23'), 'fitted' (average across all subjects), 
	    	'validation' (set used for the validation of the inference algorithm in the COIN paper),
	    	'training' (a set that maxisimises the MSE difference between the COIN inference and 
	    	a leaky integrator with a high self-transition bias), 'transitions' (a set that 
	    	produces rich transition probability matrices with a low self-transition bias), or 
	    	'transglobal' (a set with a balance between rich transition probability matrices and a 
	    	strong resemblance between the transition and global probabilities)

	    Returns
	    -------
	    dict
	       Dictionary listing the values for each hyperparameters (keys).
	"""

	if type(parset) is str:
		parsets = load_sub_pars()
		if parset in parsets:
			pars = parsets[parset]
			pars['gamma_q'] = 0.1 
			pars['gamma_t'] = 0.1 
			pars['alpha_q'] = 10. 
		else:
			pars = {}
			match parset:
				case 'fitted': # Average of evoked+spontaneous across subjects = implementation
					pars['alpha_t'] = 9.0
					pars['gamma_t'] = 0.1    # Not reported; set to coin's implementation
					pars['rho_t']   = 0.25
					pars['alpha_q'] = 25.0   # Not fitted; set to COIN implementation
					pars['gamma_q'] = 0.1    # Not fitted; set to COIN implementation
					pars['mu_a']    = 0.9425
					pars['si_a']    = 0.0012
					pars['si_d']    = 0.0008
					pars['si_q']    = 0.0089
					pars['si_r']    = 0.03   # Not reported; set to coin's implementation
				case 'validation':
					pars['alpha_t'] = 10.0
					pars['gamma_t'] = 0.1    # Not reported; set to coin's implementation
					pars['rho_t']   = 0.9
					pars['alpha_q'] = 10.
					pars['gamma_q'] = 0.1    # Not reported; set to coin's implementation
					pars['mu_a']    = 0.9
					pars['si_a']    = 0.1
					pars['si_d']    = 0.1
					pars['si_q']    = 0.1
					pars['si_r']    = 0.03   # Not reported; set to coin's implementation
				case 'training':
					pars = load_pars('validation')
					pars['gamma_t'] = 0.2
					pars['rho_t']   = 0.93
					pars['mu_a']    = 0.01
					pars['si_a']    = 0.2
					pars['si_d']    = 1.0
				case 'transitions':
					pars = load_pars('validation')
					pars['gamma_t'] = 1.5
					pars['alpha_t'] = 0.1
					pars['rho_t']   = 0.4
					pars['mu_a']    = 0.01
					pars['si_a']    = 0.2
					pars['si_d']    = 1.0
				case 'transglobal':
					pars = load_pars('validation')
					pars['gamma_t'] = 1.2
					pars['alpha_t'] = 9.0
					pars['rho_t']   = 0.6
					pars['mu_a']    = 0.01
					pars['si_a']    = 0.2
					pars['si_d']    = 1.0
		return(pars)


def load_sub_pars(types = ['S', 'E', 'F', 'V', 'M']):
	""" Loads the hyperparameter sets listed in the COIN paper

	    Parameters
	    ----------
	    types : list of strings
	    	Each string identifies a type of hyperparameters: 'S' (hyperparameters fitted to the 
	    	subjects from the spontaneous recovery experiment), 'E' (hyperparameters fitted to the 
	    	subjects from the evoked recovery experiment), 'F' (hyperparameters used to produce
	    	Fig1 from the paper), 'V' (hyperparameters used to validate the COIN inference 
	    	algorithm), 'M' (hyperparameters fitted to the subjects from the memory update
	    	 experiment)

	    Returns
	    -------
	    dict
	       Dictionary listing the hyperparametrisations (values; as dicts) for each subject (keys)
	"""

	if type(types) != list:
		types = list(types)

	with open('./subParameters.csv', 'r') as f:
		subparfile = csv.reader(f, delimiter='\t')
		fields = next(subparfile)
		subpar = dict()
		for line in subparfile:
			if line[0][0] in types:
				subpar[line[0]] = dict(  [(field, float(val)) for field, val in zip(fields[1:], line[1:])]  )

	for sub in subpar:
		subpar[sub]['si_r'] = np.round(1000 * (subpar[sub].pop('si_m')**2 + 0.03**2)**0.5) / 1000
		subpar[sub]['gamma_t'] = 1.0

	return subpar


def load_recovery_data(subs=None):
	""" Loads the data (subject responses) from the spontaneous and evoked recovery experiments 

	    Parameters
	    ----------
	    subs : str or list of str
	    	Each string identifies a subject (e.g., 'S1') or a set of subjects (e.g., 'S'). All 
	    	entries should correspond to one experiment only.

	    Returns
	    -------
	    y: np.array
	    	Subject responses
	    t: np.array
	    	Ticks identifying the trial number corresponding to each of the entries in y
	    f: np.array
	    	Field values (i.e., observations) from the experiment
	"""
	if subs == 'E':
		subs = [f'S{d}' for d in range(1,9)]
		exp = 'S'
	elif subs == 'S':
	 	subs = [f'E{d}' for d in range(1,9)]
	 	exp  = 'E'

	if type(subs) is not list:
		subs = [subs]

	exp = subs[0][0]
	if not all([exp in s for s in subs]):
		raise Exception('Trying to extract data from more than one experiment')
	
	datapath = 'adaptation_recovery.mat';
	mat = scipy.io.loadmat(datapath)

	if exp == 'S':
		y_all, t, f = mat['y_s'].T, mat['t_s'].reshape(-1)-1, mat['f_s'].reshape(-1)
	elif exp == 'E':
		y_all, t, f = mat['y_e'].T, mat['t_e'].reshape(-1)-1, mat['f_e'].reshape(-1)

	y = np.zeros((len(subs), y_all.shape[1]))
	for n, sub in enumerate(subs):
		y[n] = y_all[int(sub[1])-1]

	return y, t, f


def generate_field_sequence(experiment, noise=0.03, batch_size=1, **kwargs):
	""" Generates a set of fields (observations) corresponding to the specified experiment

	    Parameters
	    ----------
	    experiment : str
	    	Either 'evoked' (recovery), 'spontaneous' (recovery), 'savings', (retrograde) 
	    	'interference', or (effect of environmental) 'consistency' (on learning rate)
	    noise      : float
	    	standard deviation of the observation noise
	    batch_size : int
	    	number of batches

	    Returns
	    -------
	    x  : np.array
	    	sequence of observations; dim 0 runs across batches, dim 1 across time points, dim 2 
	    	is set to one for compatibility with pytorch
	"""

	match experiment:

		case 'spontaneous':
			F = [[(0, 50), (+1, 125), (-1, 15), (np.nan, 150)] for _ in range(batch_size)]

		case 'evoked':
			F = [[(0, 50), (+1, 125), (-1, 15), (+1, 2), (np.nan, 150)] for _ in range(batch_size)]

		case 'savings':
			exposure = [(0, 60), (+1, 125), (-1, 15), (np.nan, 50)]
			F = [exposure + [(0, 50)] + exposure for _ in range(batch_size)]

		case 'interference':
			Np = kwargs['Np'] 
			F = [[(0, 160), (1, Np), (-1, 115)] for _ in range(batch_size)]

		case 'consistency':
			Np, pStay = 30, kwargs['pStay']
				
			# Closure-of-block segment composed of: 2 channel trials, 10 washouts, 1 triplet
			f1 = [(np.nan, 2), (0, 10)] + [(np.nan, 1), (1, 1), (np.nan, 1)]

			F = []
			for batch in range(batch_size):
				f = [(0, 10)] #[(0, 156)]
				# Determine the duration of the +1/-1 environment fields stochastically
				for block in range(6):
					durs = [0]
					while max(durs) < 2 or min(durs) < 1 or sum(durs) != Np or len(durs) < 2:
						nChunks = (np.random.rand(Np) > pStay).sum() + 1 # n_changes + 1
						alpha = (abs(5*np.random.randn()) + 0.5,) * nChunks
						durs  = np.rint(Np*np.random.dirichlet(alpha)).astype(int)
					while sum(durs[0::2]) != sum(durs[1::2]):
						if sum(durs[0::2]) > sum(durs[1::2]):
							ix1 = np.random.choice(np.where(durs[0::2] > 1)[0], 1)[0]
							ix2 = np.random.randint(len(durs[1::2]))
							durs[0::2][ix1] -= 1
							durs[1::2][ix2] += 1
						else:
							ix1 = np.random.randint(len(durs[0::2]))
							ix2 = np.random.choice(np.where(durs[1::2] > 1)[0], 1)[0]
							durs[0::2][ix1] += 1
							durs[1::2][ix2] -= 1
					n0 = np.random.randint(2)
					f += [((-1)**(n+n0), dur) for n, dur in enumerate(durs)] + f1
				F.append(f)

	fields = np.array([np.concatenate([[s[0]] * s[1] for s in f]) for f in F])
	fields = fields + noise * np.random.randn(*fields.shape)
	x = fields[:,:,None]

	return x


def initialise_experiments_U(N, experiments=None):
	""" Auxiliary function that initialises a dictionary to store predictions corresponding to one 
		or several of the experiments. To be called by model-specific functions.

	    Parameters
	    ----------
	    N : int
	    	number of runs/batches
	    experiments : list of str (optional)
			Lists the required experiments; if not provided it assumes all experiments are required

	    Returns
	    -------
	    U0: dict
	    	Each entry is a dict initialising the dict-fields required to store the results for 
	    	each experiment (keys)
	"""
	U0 = {'spontaneous': {'data': np.zeros((N, 340))},
	      'evoked': {'data': np.zeros((N, 342))},
	      'savings': dict([(k, np.zeros((N, 125))) for k in ['first', 'second']]),
	      'interference': dict([(n, np.zeros((N, 115))) for n in [0, 13, 41, 112, 230, 369]]),
	      'consistency': dict([(p, np.zeros((N, 6))) for p in [0.1, 0.5, 0.9]])}

	if experiments is not None:
		for exp in [k for k in U0 if k not in experiments]:
			del(U0[exp])

	return U0


def compute_model_avg_mse(U):
	""" Auxiliary function that computes the average and mse of the predictions across several runs
		of the experiments of the COIN paper.

	    Parameters
	    ----------
	    U : dict()
	    	A dictionary encoding the predictions across several batches; same structure as produced
	    	by initialise_experiments_U

	    Returns
	    -------
	    model_avg: dict
	    	Each entry is a [dict storing the average for each of the conditions (keys)] for each 
	    	experiment (keys)
	    mse: dict
	    	Each entry is a float with the average mse across conditions and trials for each 
	    	experiment (keys)

	"""

	dd = load_group_data()
	model_avg, mse = dict(), dict()
	
	for exp in U:
		model_avg[exp], mse[exp] = dict(), 0
		for key in dd[exp]:
			model_avg[exp][key] = U[exp][key].mean(0) 
			preds = model_avg[exp][key][dd[exp][key]['xticks'].astype(int)]
			mse[exp] += ((dd[exp][key]['avg'] - preds)**2).mean() / len(dd[exp])

	return model_avg, mse


def estimate_subject_fits(subs=None):
	""" Estimates the predictions of the COIN inference model across a set of subjects for all the
		experiments from the COIN paper that do not involve cue emissions. 

	    Parameters
	    ----------
	    subs : str or list of str (optional)
	    	Each string identifies a subject (e.g., 'S1') or a set of subjects (e.g., 'S'). If not
	    	specified it uses all 'S' and 'E' subjects.

	    Returns
	    -------
	    U : dict()
	    	A dictionary encoding the predictions across subjects and experiments; same structure 
	    	as produced by initialise_experiments_U
	"""

	if subs is None:
		subs = ['S', 'E']

	subs = list(load_sub_pars(subs).keys())

	eng = matlab.engine.start_matlab()
	eng.addCoinPaths(nargout=0)

	U = initialise_experiments_U(len(subs))

	for s, sub in enumerate(subs):
		Usub, X0 = CRFGenerativeModel(sub).all_coin_experiments(eng=eng)
		for key in Usub:
			if type(Usub[key]) is dict:
				for subkey in Usub[key]:
					U[key][subkey][s] = Usub[key][subkey].mean(0)
			else:
				U[key]['data'][s] = Usub[key].mean(0)

	return U


def plot_group_avg(U=None, modname=None, modmap="Greens", ddmap="Purples", axs=None, savefig=None):
	""" Plots a set of predictions across all experiments from the COIN paper that do not involve
		cue emissions, optionally against the actual experimental data. 

	    Parameters
	    ----------
	    U       : dict() (optional)
	    	A dictionary encoding the predictions across subjects and experiments; same structure 
	    	as produced by initialise_experiments_U
		modname : str (optional)
			String identifying the model that produced the predictions; if not specified it assumes
			COIN
		modmap  : str or None (optional)
			mycolorpy.colorlistp colormap (e.g., 'Blues', 'Greens', etc) to use for plotting the
			model predictions. If not specified it assumes 'Greens'; if None it does not plot the
			model predictions.
		ddmap   : str or None (optional)
			mycolorpy.colorlistp colormap (e.g., 'Blues', 'Greens', etc) to use for plotting the
			data. If not specified it assumes 'Purples'; if None it does not plot the data.
		axs     : (optional) an np.array instance of plt.subplots()
			plt axes where to plot the results
		savefig : (optional) str or None
			filepath (without extension) to save the produced figure; if None or not specified it
			does not print the figure (it can be shown with plt.show())
	    
	    Returns
	    -------
		dict (only if modmap and U are not None; i.e., only if plotting data) 
			mse of the predictions with respect to the data for all experiments (keys)

	"""

	if U is None and modmap is not None:
		U = estimate_subject_fits()
		modname = 'COIN'

	if modmap is not None:
		model_avg, mse = compute_model_avg_mse(U)
	
	if ddmap is not None:
		dd = load_group_data()

	if axs is None:
		fig, axs = plt.subplots(1, len((U if U is not None else dd)))
	else:
		fig = None

	ax = dict([(exp, axs[n]) for n, exp in enumerate((U.keys() if U is not None else dd.keys()))])
	jointplot = (modmap is not None and ddmap is not None)

	for exp in ax:
		if modmap is not None:
			d = plot_experiment(U[exp], exp, ax[exp], modmap, legsuffix = modname if jointplot else '')
		if ddmap is not None:
			plot_experiment(dd[exp], exp, ax[exp], ddmap, legsuffix = 'data' if jointplot else '')

	if not jointplot:
		color = f'tab:{(modmap if ddmap is None else ddmap).lower()[:-1]}' 
		txtspec = dict(ha='left', va='center', fontsize=20, color=color)
		plt.text(-0.2, 0.5, modname, rotation=90, transform=axs[0].transAxes, **txtspec)

	if savefig is not None and fig is not None:
		fig.subplots_adjust(left=0.06,right=0.98,bottom=0.17,top=0.9,wspace=0.3,hspace=0.3)
		fig.set_size_inches(20, 3)
		plt.savefig(f'{savefig}.png')

	if modmap is not None:
		return mse


def plot_experiment(u, experiment, axs=None, cmap="Blues", legsuffix=''):
	""" Plots a set of predictions across all experiments from the COIN paper that do not involve
		cue emissions, optionally against the actual experimental data. 

	    Parameters
	    ----------
	    u  : dict() (optional)
	    	A dictionary encoding the predictions across the experiment conditions (keys); same 
	    	structure as each of the sub-dicts of U produced by initialise_experiments_U
		experiment : str 
			name of the experiment corresponding to the provided predictions; should be one of 
			the experiments considered in generate_field_sequence
		axs  : (optional) an instance of plt.subplots()
			plt axes where to plot the predictions
		cmap  : str or None (optional)
			mycolorpy.colorlistp colormap (e.g., 'Blues', 'Greens', etc) to use for plotting the
			predictions
		legsuffix : (optional) str or None
			suffix for the legends of the provided predictions; useful for plotting several sets
			of predictions in the same axes 
	"""

	if axs is None:
		fig, axs = plt.subplots(1,1)

	match experiment:
		case 'consistency':
			labels = dict([(key, f'pstay = {key:.1f} {legsuffix}') for key in u])
			xlabel = 'block number'
			ylabel = 'single trial learning'
		case 'interference':
			labels = dict([(key, f'N+ = {key:d} {legsuffix}') for key in u])
			xlabel = 'trials after field change'
			ylabel = 'field prediction'
		case 'savings':
			labels = dict([(key, f'{key} encounter {legsuffix}') for key in u])
			xlabel = 'trials after field change'
			ylabel = 'field prediction'
		case 'spontaneous' | 'evoked':
			labels = {'data': None if legsuffix == '' else legsuffix}
			xlabel = 'trial number'
			ylabel = 'field prediction'	

	colours = dict([(k, c) for k, c in zip(u, mcp.gen_color(cmap, len(u)+1, True)[:-1])])

	for key in u:
		if type(u[key]) is dict:
			xt   = u[key]['xticks']
			rAvg = u[key]['avg']
			rStd = u[key]['sem']
		else:
			xt = range(u[key].shape[1]) if len(u[key].shape) > 1 else range(u[key].shape[0])
			rAvg = u[key].mean(0) if len(u[key].shape) > 1 else u[key]
			rStd = u[key].std(0) / np.sqrt(u[key].shape[0]) if len(u[key].shape) > 1 else None
		
		axs.plot(xt, rAvg, color=f'{colours[key]}', label=labels[key])
		if rStd is not None:
			axs.fill_between(xt, rAvg-rStd, rAvg+rStd, color=f'{colours[key]}',  alpha=0.2)

	axs.set_xlabel(xlabel)
	axs.set_ylabel(ylabel)
	axs.set_title(experiment)
	if len(labels) > 1 or legsuffix != '': 
		axs.legend()


def plot_predictions(x, c=None, u=None, s=None, ax=[]):
	""" Auxiliary function that plots a single line of observations/fields, an average prediction,
		and its associated uncertainty. 

	    Parameters
	    ----------
	    x  : np.array
	    	A one-dimensional array encoding the line of observations/fields
		c  : np.array (optional)
	    	A one-dimensional array encoding the contexts corresponding to the fields; when 
	    	specified, observations are colour-coded corresponding to their context 
		u  : np.array (optional)
			A one-dimensional array encoding the average prediction
		s  : np.array (optional)
			A one-dimensional array encoding the uncertainty on the prediction (e.g., std)
		axs  : (optional) an instance of plt.subplots()
			plt axes where to plot the prediction
	"""

	if ax == []:
		ax = plt.axes()

	if u is not None:
		ax.plot(range(len(x)), u, color='k')
	if s is not None:
		ax.fill_between(range(len(x)), u-s, u+s, color='k', alpha=0.2)

	if c is None:
		ax.plot(range(len(x)), x, color='tab:blue')
	else:
		nColCycles = int(np.ceil((max(c) + 1) / 10))
		colours = list(matplotlib.colors.TABLEAU_COLORS.values()) * nColCycles
		time = np.arange(len(x), dtype=int)

		T = [0] + [t for t in range(1, len(x)) if (c[t]-c[t-1])!=0] + [len(x)]
		for t0, t1 in zip(T[:-1], T[1:]):
			ax.plot(time[t0:t1], x[t0:t1], color = colours[c[t0]])

	ax.set_xlabel('trial number')
	ax.set_ylabel('field / output')


def load_group_data():
	""" Auxiliary function that loads the behavioural group data for the five experiments of the 
		COIN paper that do not involve cue emissions

	    Returns
	    ----------
	    dict
			A dictionary with the data for each of the experiments (keys). Values are dictionaries
			with entries for each experimental condition (keys). Values in each condition are 
			dictionaries encoding the (keys) mean, std, and timepoints of the measurements.

	"""

	with open('./expsgroupdata.pickle', 'rb') as f:
		dd = pickle.load(f)

	return(dd)



