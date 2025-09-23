import collections.abc
import copy
import functools
import multiprocessing
import numpy as np
import os.path
import pathlib
import pickle
import random
import scipy.optimize
import scipy.special
import scipy.stats as ss
import sys
import time

from icecream import ic
from pathos.multiprocessing import ProcessingPool
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'COIN_Python')))
import COIN_Python.coin as coinp

class GenerativeModel():
    """
    Core COIN Generative Model class for context-dependent inference.

    Attributes
    ----------
    parset : str or dict
        as str it identifies a hyperparametrisation compatible with load_pars()
        as dict it should have fields:
            parset['name']: name to identify the hyperparametrisation
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

    estimate_coin(y, nruns, n_ctx, max_cores)
        Runs the COIN inference algorithm using Python implementation and returns the predictions
        for the observations hat{y}_t, the log-probabilities of the actual input sequence y_t,
        the cumulative probabilities of y_t, and the responsibilities of the contexts lambda^c_t

    estimate_leaky_average(y, tau)
        Runs a leaky integrator with integration time constant tau to generate predictions for the
        observations hat{y}_t on the input sequence y_t. Returns the predictions for the
        observations hat{y}_t, the log-probabilities of the actual input sequence y_t, and the
        cumulative probabilities of y_t, and tau. If tau is not specified it estimates the best
        value for the current hyperparametrisation.

    fit_best_tau(n_trials)
        Finds the integration time constant tau minimising prediction error for the current
        hyperparametrisation and number of trials.

    benchmark(n_trials, n_instances, suffix, save)
        Performs a thorough benchmarking of the generative model for the given number of trials.
        It returns a dictionary indicating the performance of the COIN generative model and an
        optimal leaky integrator, and the context and observation sequences used to generate
        the benchmarks. The function stores the benchmarks in a pickle for easy retrieval and only
        performs the computations if the pickle file does not exist.

    measure_ctx_transition_stats(n_trials)
        Measures the empirical number of visited contexts, context transition probability matrix,
        and context global probabilities for sequences with n_trials timepoints and the current
        hyperparametrisation.

    """

    def __init__(self, parset):

        self.pool = ProcessingPool()

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
        self.max_cores = self.pars['max_cores']


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

    # Auxiliary samplers
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

    # Main data sampler
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
            sequence of state observations; dim 0 runs across batches, dim 1 across time points, dim 2 
            is set to one for compatibility with pytorch
        q  : np.array
            sequence of cues; same dimensional arrangement as y. Cue emissions are untested.
        c  : np.array 
            sequence of sampled contexts; same dimensional arrangement as y. 
        """

        self.n_trials = n_trials # hack to avoid passing multiple parameters to pool.map

        # Next line ensures all instances are sampled with different seeds
        seeds = np.random.randint(low=1, high=1024*16, size=(batch_size)).cumsum()
        
        if self.max_cores is None or self.max_cores > 1:
            pool = ProcessingPool(nodes=self.max_cores)
            res = pool.map(self.generate_session, seeds)
            # res = self.pool.map(self.generate_session, seeds)
        else:
            res = [self.generate_session(seed) for seed in seeds]

        # parallel_generate_session = functools.partial(self.generate_session)
        # with multiprocessing.Pool() as pool:
        #     res = pool.map(parallel_generate_session, seeds)


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

        if self.max_cores is None or self.max_cores > 1:
            # res = self.pool.map(self.sample_contexts, seeds)
            pool = ProcessingPool(nodes=self.max_cores)
            res = pool.map(self.sample_contexts, seeds)
        else:
            res = [self.sample_contexts(seed) for seed in seeds]

        # parallel_generate_session = functools.partial(self.generate_session)
        # with multiprocessing.Pool() as pool:
        #     res = pool.map(parallel_generate_session, seeds)

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
        """Generates a single data sequence y_t given a sequence of contexts c_t and a sequence of 
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

    def sample_states(self, contexts, return_pars=False):
        """Generates a single data sequence y_t given a sequence of contexts c_t a sequence of 
        states x_t^c

        Parameters
        ----------
        contexts : integer np.array
            one-dimensional sequence of contexts 
        return_pars: bool
            also returns the retention and drift parameters for each context


        Returns
        -------
        states : dict
            dictionary encoding the latent state values (one-dimensional np.array) for each 
            context c (keys).
        a: retention parameters for each context (only if return_pars set to True)
        d: drift parameters for each context (only if return_pars set to True)

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

        if return_pars:
            return states, a, d
        else:
            return states

    def order_contexts(self, contexts):
        """Reorders a list of contexts so that they appear in ascending order

        Parameters
        ----------
        contexts : list of arrays
            sequence of contexts 

        Returns
        -------
        ordered_contexts : list of arrays
            sequence of contexts ordered in ascending order
        
        """

        c2c = dict()
        ordered_contexts = np.zeros(self.n_trials, dtype=int)
        next_c = 0

        for t in range(len(contexts)):
            if contexts[t] not in c2c.keys():
                c2c[contexts[t]] = next_c
                next_c += 1
            ordered_contexts[t] = c2c[contexts[t]]

        return list(ordered_contexts)

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

        Returns.
        -------
        contexts : list or integers
            sequence of sampled contexts
        """

        if seed is not None:
            np.random.seed(seed)
        if n_trials is not None: 
            self.n_trials = n_trials

        kappa_t = self.alpha_t * self.rho_t / (1 - self.rho_t)
        N = np.zeros((10, 10))  # total N context transitions
        
        contexts, N[0, 0] = [0], 1  # Initialisation (assuming contexts[0] = 0)
        beta = self._break_new_partition_([1], self.gamma_t)

        for t in range(1, self.n_trials):
            # Sample context
            contexts.append(self._sample_customer_(beta, N, contexts[t-1], self.alpha_t, kappa_t))
            if contexts[-1] == len(beta)-1: # If opening a new context:
                beta = self._break_new_partition_(beta, self.gamma_t)
                if len(beta) > N.shape[0]: # pad N if contexts > max_ctx
                    N = np.pad(N, ((0, N.shape[0]), (0, N.shape[0])))
            N[contexts[-2], contexts[-1]] += 1  # Add transition count 

        return contexts

    def sample_cues(self, contexts):
        """Samples a sequence of cues. Each call uses an independent sample of pi_q and beta_q.
        Cue emission has not been thoroughly tested

        Parameters
        ----------
        contexts : list of integers
            sequence of sampled contexts; dim 0 runs across time points, dim 1 is set to one

        Returns
        -------
        cues : list of integers
            sequence of emitted cues
        """

        N = np.zeros((max(contexts) + 1, 10)) # total N context ~ cue pairs
        
        cues, N[0, 0] = [0], 1 # Initialisation (assuming cues[0] = 0)
        beta = self._break_new_partition_([1], self.gamma_q)

        for c_t in contexts[1:]:
            cues.append(self._sample_customer_(beta, N, c_t, self.alpha_q)) # Sample cue
            if cues[-1] == len(beta): # If opening a new cue:
                beta = self._break_new_partition_(beta, self.gamma_q) 
                if cues[-1] > N.shape[0] - 1: # pad N if cues > max_cues
                    N = np.pad(N, ((0, N.shape[0]), (0, 0)))
            N[c_t, cues[-1]] += 1  # Add cue-context association count 
            
        return cues

    # Coin estimation
    def estimate_coin(self, y, nruns=1, n_ctx=10, max_cores=1):
        """Runs the COIN inference algorithm on a batch of observations using the Python implementation

        Parameters
        ----------
        y   : np.array
            sequence of observations; dim 0 runs across batches, dim 1 across time points, dim 2 is
            set to one
        nruns : int, optional
            number of COIN runs (default: 1)
        n_ctx : int, optional
            maximum number of contexts (default: 10)
        max_cores : int, optional
            maximum number of cores for parallel processing (default: 1)

        Returns
        -------
        z_coin : np.array
            predictions hat{y}_t for the observations y_{1:t-1}; same dimensional arrangement as y (n_samples, n_trials)
        logp   : np.array
            log-probabilities of the input sequence y_t under the COIN posterior distribution; shape = (n_samples, nruns)
        cump   : np.array
            cumulative probabilities of the input sequence y_t under the COIN posterior
            distribution; same dimensional arrangement as y. Useful to measure calibration.
        lamb   : np.array
            responsibilities lambda^c_t for each context c and time-step t. dim 0 runs across
            batches, dim 2 across time points, dim 1 across contexts (dimension equals the maximum
            number of contexts of the COIN model, currently set to 10+1)

        """

        # Translation to the naming of the hyperparameters in the COIN inference algorithm
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

        z_coin, logp, cump, lamb, _, _ = runCOIN(y, parlist, parvals, nruns=nruns, n_ctx=n_ctx, max_cores=max_cores)

        return(z_coin, logp, cump, lamb)


    # Baselines and heuristic estimations
    def theoretical_expected_beta(self, n_contexts=11):
        """Computes the theoretical expected value for the global distribution assuming contexts
        are sampled from distributions beta ~ GEM(gamma_t). This distribution only matches the 
        global distribution of contexts for the explicit method.

        Parameters
        ----------
        n_contexts : int (optional)
            Maximum number of contexts considered

        Returns
        -------
        e_beta : np.array
            E[beta] distribution up to item n_contexts (note that e_beta.sum()<1)
        """
        e_beta = [((self.gamma_t)**j) / ((1+self.gamma_t)**(j+1)) for j in range(n_contexts)]
        return np.array(e_beta)

    def empirical_expected_beta(self, n_contexts=11, n_trials=1000, n_samples=1000):
        """Computes the empirical expected value for the global distribution of contexts.

        Parameters
        ----------
        n_contexts : int (optional)
            Maximum number of contexts considered

        Returns
        -------
        e_beta : np.array
            E[beta] distribution up to item n_contexts (note that e_beta.sum()<1)
        """
        c_series  = self.generate_context_batch(n_trials, n_samples)['C']

        # Empirical distribution of contexts in each sample
        empirical = [[(c == ctx).mean() for ctx in range(n_contexts)] for c in c_series]

        # Single sample of uniform distribution for contexts with p < 1/n_samples
        uniform   = [1/n_contexts for ctx in range(n_contexts)]

        # Global empirical distribution
        glob_dist = np.array(empirical + [uniform]).mean(0)

        return glob_dist

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
            predictions hat{y}_t for the observations y_{1:t-1}; same dimensional arrangement as y (n_samples, n_trials)
        logp   : np.array
            log-probabilities of the input sequence y_t; for compatibility with COIN inference, averaged across time-points hence of shape (n_samples, 1)
        cump   : np.array
            cumulative probabilities of the input sequence y_t. Useful to measure calibration.

        """
        if tau is None:
            if not hasattr(self, 'best_t'):
                self.fit_best_tau(n_trials = y.shape[1])# However fit_best_tau should be called before so as to avoid nested processing pool 
            tau = self.best_t

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

        logp = -0.5 * np.log(2*np.pi) -np.log(s_slid) - 0.5 * ((z_slid - y[:,:,np.newaxis]) / s_slid)**2
        # logp = logp.mean(1) # TODO: check # NOTE: this gives the joint log p of all observations
        cump = 0.5 * (1 + scipy.special.erf((y[:,:,np.newaxis] - z_slid) / (np.sqrt(2) * s_slid)))

        if len(tau) == 1:
            z_slid, logp, cump = z_slid[..., 0], logp[..., 0], cump[..., 0]

        return(z_slid, logp, cump)

    def _estimate_leaky_average_call_(self, inpars):
        """ Wrapper of estimate_leaky_average to take in multiple parameters during 
        parallelisation; not meant as a user-end method"""

        x, tau = inpars
        return(self.estimate_leaky_average(x, tau=tau))
    
    def estimate_leaky_average_parallel(self, X, tau=None):
        """Call estimate_leaky_average with multiprocessing pool wrapper"""

        parallel_function = functools.partial(self.estimate_leaky_average)
        with multiprocessing.Pool() as pool: # multiprocessing.Pool()
           res = pool.starmap(parallel_function, [(x[None, ...], tau) for x in X])

        # res = self.pool.map(self._estimate_leaky_average_call_, [(x[None, ...], tau) for x in X])

        temp = res
        
        z_slid    = np.array([r[0][0, :] for r in temp])
        logp    = np.array([r[1][0, :] for r in temp])
        cump = np.array([r[2][0, :] for r in temp])
        
        return(z_slid, logp, cump)

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

        X = self.generate_batch(n_trials, n_train_instances)[0][:, :, 0]

        def fn(tau):
            if self.max_cores is None or self.max_cores > 1:
                pool = ProcessingPool(nodes=self.max_cores)
                res = pool.map(self._estimate_leaky_average_call_, [(x[None, ...], tau) for x in X])
                # res = self.pool.map(self._estimate_leaky_average_call_, [(x[None, ...], tau) for x in X])
            else:
                res = [self.estimate_leaky_average(x[None, ...], tau) for x in X]
            mse = np.mean([((r[0] - x)**2).mean() for r, x in zip(res, X)])
            return(mse)


        optimRes = scipy.optimize.minimize_scalar(fn, bounds=(0, X.shape[1]), 
                                                     method='bounded', 
                                                     options={'xatol':1e-3})				
        if optimRes.success:
            best_t = optimRes.x
        else:
            taus = np.arange(100, -1, -1)
            if self.max_cores is None or self.max_cores > 1:
                pool = ProcessingPool(nodes=self.max_cores)
                res  = pool.map(self._estimate_leaky_average_call_, [(x, taus) for x in X])
                # res  = self.pool.map(self._estimate_leaky_average_call_, [(x, taus) for x in X])
            else:
                res = [self.estimate_leaky_average(x, tau=taus) for x in X]         
            
            z_slid = np.array([r[0] for r in res])
            best_t = taus[((z_slid[:, 0, :] - X)**2).mean((0,1)).argmin()]

        self.best_t = best_t

        return best_t

    # Example plotting

    # Benchmarks
    def benchmark(self, n_trials=1000, n_instances=16, suffix=None, save=True):
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
            minloglik = -4.6

            print(f'### Computing benchmarks for {self.parset} ###')
            t0 = time.time()

            print(f'Finding best tau...', end=' ', flush=True)
            tau = self.fit_best_tau(n_trials)
            print(f'[best_t = {tau:.1f}]', end=' ', flush=True)

            print(f'Generating data...', end=' ', flush=True)
            X, Q, C = self.generate_batch(n_trials, n_instances)

            print(f'Estimating LI...', end=' ', flush=True)
            res = self.pool.map(self._estimate_leaky_average_call_, [(x[None, :, 0], tau) for x in X]) # NOTE: probably to be replaced with x[None, ...] or even just x

            z_slid    = np.array([r[0][0, :] for r in res])
            p_slid    = np.array([r[1][0, :] for r in res])
            cump_slid = np.array([r[2][0, :] for r in res])

            F = np.linspace(0, 1, 1000)
            cums_cump_slid = np.array([(cump_slid <= f).sum(1)/n_trials for f in F])
            LI_mse = ((z_slid - X[..., 0])**2).mean(1)
            LI_kol = abs(cums_cump_slid - F[:, None]).max(0)
            LI_ce  = p_slid.mean(1) # Observations cross entropy
            LI_ct_ac = (C[:, :, 0] == 0).mean(1) # LI assumed to predict always context 0 # Context identification accuracy
            LI_ct_p  = (C[:, :, 0] == 0).mean(1) # Probability of the actual context on the posterior of the prediction
            # p_ctx not defined in LI --> we predict instead the empirical global distribution across c
            e_beta = self.empirical_expected_beta(n_trials=n_trials, n_samples=100*n_instances) 
            LI_ct_ce = np.zeros(C.shape[0])
            for b in range(C.shape[0]):
                for ctx in range(len(e_beta)):
                    LI_ct_ce[b] += np.log(e_beta[ctx]) * (C[b, :, 0] == ctx).mean()

            print(f'Estimating coin...', flush=True)

            z_coin, ll_coin, cump_coin, lamb = self.estimate_coin(X, n_ctx=64)
            loglamb = np.log(lamb + np.exp(minloglik))

            coin_mse = ((z_coin - X[..., 0])**2).mean(1)
            cums_cump_coin = np.array([(cump_coin[..., 0] <= f).sum(1)/cump_coin.shape[1] for f in F])
            coin_kol = abs(cums_cump_coin - F[:, np.newaxis]).max(0)
            coin_ce  = ll_coin.mean(1)

            c_hat = np.argmax(lamb, axis=1) # Predicted context
            coin_ct_ac = (c_hat == C[..., 0]).mean(1) # Context identification accuracy
            coin_ct_p  = np.zeros(C.shape[0]) # Probability of the actual context on the posterior of the prediction
            coin_ct_ce = np.zeros(C.shape[0]) # Context cross-entropy

            # For each sample:
            for b in range(C.shape[0]):
                # For each context:
                for ctx in range(lamb.shape[1]):
                    # Get timepoints where predicted context was correct (indices of hits)
                    ctx_ix = np.where(C[b, :, 0] ==ctx)[0]
                    # Compute cumulative probability over time points
                    coin_ct_p[b] += np.nansum(lamb[b, ctx, ctx_ix]) / C.shape[1]
                    loglambs = np.maximum(minloglik, np.log(lamb[b, ctx, ctx_ix]))
                    coin_ct_ce[b] += np.nansum(loglambs) / C.shape[1]


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
            print(f'mse_slid/mse_coin = {rat:.2f}', end='; ')
            print(f'lopg_ctx_coin - logp_ctx_slid = {dif:.2f}')

            benchmark_kit = {'X': X, 'C': C, 'perf': perf, 'best_t': tau}

            if not os.path.exists(os.path.split(benchmarkpath)[0]):
                os.mkdir(os.path.split(benchmarkpath)[0])

            if save:
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

    # Auxiliary functions
    def _break_new_partition_(self, beta, gamma):
        """Runs one stick-breaking step of the GEM distribution

        Parameters
        ----------
        beta : list
            Partition of a stick of measure 1 (sum(beta) = 1); last item corresponds
             to the measure that has not yet been assigned

        Returns
        -------
        beta: list
           Partition of a stick of measure 1 with one more partition (len(beta) 
           increased by one)
        """

        w = ss.beta.rvs(1, gamma) # Stick-breaking weight
        beta.append((1-w) * beta[-1])
        beta[-2] = w * beta[-2]
        return beta

    def _sample_customer_(self, beta, N, j, alpha, kappa=0):
        """Samples one customer from a CRF-like discrete distribution with or without
        a self-transition bias

        Parameters
        ----------
        beta : list
            Partition of a stick of measure 1 (sum(beta) = 1); last item corresponds
             to the measure that has not yet been assigned
        N    : 
            Customer-table counts; N[j, :] corresponds to the counts for the current
            restaurant j
        j    : 	
            Current restaurant
        kappa: 
            Self-transition bias (set to 0 for a non-sticky process)


        Returns
        -------
        beta: list
           Partition of a stick of measure 1 with one more partition (len(beta) 
           increased by one)
        """
        beta_w   = alpha * np.array(beta)
        sticky_w = 0 if kappa == 0 else kappa * np.eye(len(beta))[j]
        global_w = N[j, 0:len(beta)]
        return random.choices(range(len(beta)), beta_w + sticky_w + global_w, k=1)[0]


def load_pars(parset):
    """ Loads a set of COIN hyperparameters from a pre-specified label.

        Parameters
        ----------
        parset : str
            String identifying the hyperparametrisation. Can be: 'fitted' (average across all subjects),
            'validation' (set used for the validation of the inference algorithm in the COIN paper),
            'training' (a set that maximises the MSE difference between the COIN inference and
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
        pars = {}
        if parset == 'fitted': # Average of evoked+spontaneous across subjects = implementation
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
        elif parset == 'validation':
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
        elif parset == 'training':
            pars = load_pars('validation')
            pars['gamma_t'] = 0.2
            pars['rho_t']   = 0.93
            pars['mu_a']    = 0.01
            pars['si_a']    = 0.2
            pars['si_d']    = 1.0
        elif parset == 'transitions':
            pars = load_pars('validation')
            pars['gamma_t'] = 1.5
            pars['alpha_t'] = 0.1
            pars['rho_t']   = 0.4
            pars['mu_a']    = 0.01
            pars['si_a']    = 0.2
            pars['si_d']    = 1.0
        elif parset == 'transglobal':
            pars = load_pars('validation')
            pars['gamma_t'] = 1.2
            pars['alpha_t'] = 9.0
            pars['rho_t']   = 0.6
            pars['mu_a']    = 0.01
            pars['si_a']    = 0.2
            pars['si_d']    = 1.0
        
        pars['max_cores'] = 12

        return(pars)







def instantiate_coin(parlist, parvals):
    inf = coinp.COIN()
    
    # Set coin inference parameters according to user passed parameters list and values
    for i in range(len(parlist)):
        setattr(inf, parlist[i], float(parvals[i]))

    inf.store = ['predicted_probabilities', 'state_var', 'state_mean', 'drift', 'retention', 'average_state']
    inf.particles = 100
    inf.sigma_motor_noise = 0
    
    return inf
        

def call_coin(y, parlist=[], parvals=[], nruns=10, n_ctx=10, max_cores=1):
    y = np.squeeze(y)       
    
    inf = instantiate_coin(parlist, parvals)
    
    inf.runs = nruns
    inf.max_contexts = n_ctx
    inf.max_cores = max_cores

    # inf.max_cores = nruns # NOTE: Translated from runCOIN.m but probably wrong in Matlab script 
    inf.perturbations = y

    out  = inf.simulate_coin()
    mu   = np.zeros((len(y), nruns))
    cump = []
    # Original
    # logp = np.zeros((nruns, 1))
    # Suggested
    logp = np.zeros((len(y), nruns))
    lamb = np.zeros((inf.max_contexts + 1, len(y), nruns))
    a    = np.zeros((inf.max_contexts + 1, len(y), nruns))
    d    = np.zeros((inf.max_contexts + 1, len(y), nruns))
    
    # Average results over the nruns runs
    for i in range(nruns): 
        mu_parts = out["runs"][i]["state_mean"]
        sigma_parts = np.sqrt(out["runs"][i]["state_var"] + inf.sigma_sensory_noise**2)
        lambda_parts = out["runs"][i]["predicted_probabilities"] # Predicted probabilities of contexts, shape: (max_contexts, self.particles, n_trials) (max_contexts = 10+1, self.particles = 100)
        
        # Predicted observations
        mu[:, i] = np.reshape(np.mean(np.sum(lambda_parts * mu_parts, axis=0), axis=0), (1, len(y)))
        
        # True observations
        y_parts = np.tile(y, (mu_parts.shape[0], mu_parts.shape[1], 1))
        
        # Probability (Gaussian likelihood function)  of true y_i to be in the predictive (normal) distribution of the states
        p_y_parts = np.sum((lambda_parts / (np.sqrt(2 * np.pi) * sigma_parts)) * 
                        np.exp(-(y_parts - mu_parts)**2 / (2 * sigma_parts**2)), axis=0) # summed over particles?
        
        # Overal logp value across n_trials, original formulation:
        # logp[i] = np.mean(np.log(np.maximum(np.mean(p_y_parts, axis=0), np.finfo(float).eps))) # First, average over particles, then multiply (i.e., sum the logs) over trials ---> so sum, not mean
        
        # Suggestd formulation:
        # logp[i] = np.sum(np.log(np.maximum(np.mean(p_y_parts, axis=0), np.finfo(float).eps))) # First, average over particles, then multiply (i.e., sum the logs) over trials ---> so sum, not mean
        
        # Preserving individual time point (trials) values:
        logp[:, i] = np.log(np.maximum(np.mean(p_y_parts, axis=0), np.finfo(float).eps)) # First, average over particles, then multiply (i.e., sum the logs) over trials ---> so sum, not mean
        lamb[:, :, i] = np.reshape(np.mean(lambda_parts, axis=1), (lambda_parts.shape[0], len(y)))
    
        a[:, :, i] = np.reshape(np.mean(out["runs"][i]["retention"], axis=1), (lambda_parts.shape[0], len(y)))
        d[:, :, i] = np.reshape(np.mean(out["runs"][i]["drift"], axis=1), (lambda_parts.shape[0], len(y)))
        
        # Cumulative distribution of the normal distribution
        cump.append(np.reshape(np.mean(np.sum(lambda_parts * 0.5 * (1 + scipy.special.erf((y_parts - mu_parts) / (np.sqrt(2)
                                                * sigma_parts))), axis=0), axis=0), (len(y), 1)))
        
    
    mu_ = np.mean(mu, axis=1)
    # Initially: only 1 value
    # logp_ = np.max(logp) + np.log(np.sum(np.exp(logp - np.max(logp)))) - np.log(nruns)
    # To obtain (n_trials, ) like for mu_:
    logp_ = np.max(logp, axis=1) + np.log(np.sum(np.exp(logp - np.max(logp, axis=1)[:, np.newaxis]), axis=1)) - np.log(nruns)
    lamb_ = np.mean(lamb, axis=2)
    a_ = np.mean(a, axis=2)
    d_ = np.mean(d, axis=2)
    
    cump_ = np.concatenate(cump)

    return mu_, logp_, cump_, lamb_, a_, d_


def runCOIN(y, parlist, parvals, nruns=10, n_ctx=10, max_cores=1):
    """_summary_

    Args:
        y (_type_): state observations
        parlist (_type_): _description_
        parvals (_type_): _description_


    Returns:
        mu (z_coin) : np.array
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

    mu, logp, cump, lamb, a, d = [], [], [], [], [], []
    
    for b in tqdm(range(y.shape[0])):
        mu_b, logp_b, cump_b, lamb_b, a_b, d_b = call_coin(y[b, :], parlist, parvals, nruns=nruns, n_ctx=n_ctx, max_cores=max_cores)
        mu.append(mu_b)
        logp.append(logp_b)
        cump.append(cump_b)
        lamb.append(lamb_b)
        a.append(a_b)
        d.append(d_b)

    mu = np.array(mu)
    logp = np.array(logp)
    cump = np.array(cump)
    lamb = np.array(lamb)
    a = np.array(a)
    d = np.array(d)
    
    return mu, logp, cump, lamb, a, d