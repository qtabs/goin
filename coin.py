import copy
import numpy as np
import os.path
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
import inference_utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'COIN_Python')))
import COIN_Python.coin as coinp

class COINInference:
    """COIN inference engine with integrated configuration for context-dependent time series analysis."""

    def __init__(self, generative_params, particles=100, sigma_motor_noise=0.0, store_fields=None):
        # Configuration
        self.particles = particles
        self.sigma_motor_noise = sigma_motor_noise
        self.store_fields = store_fields or [
            'predicted_probabilities', 'state_var', 'state_mean',
            'drift', 'retention', 'average_state'
        ]

        # Parameters
        self.params = generative_params
        self._parameter_mapping = self._create_parameter_mapping()

    def _create_parameter_mapping(self):
        """Create mapping from internal to COIN algorithm parameters."""
        return {
            'si_q': 'sigma_process_noise',
            'si_r': 'sigma_sensory_noise',
            'mu_a': 'prior_mean_retention',
            'si_a': 'prior_precision_retention',
            'si_d': 'prior_precision_drift',
            'gamma_t': 'gamma_context',
            'alpha_t': 'alpha_context',
            'rho_t': 'rho_context',
            'gamma_q': 'gamma_cue',
            'alpha_q': 'alpha_cue'
        }

    def _setup_coin_parameters(self):
        """Convert internal parameters to COIN format with precision conversion."""
        parlist, parvals = [], []
        for param_name, param_value in self.params.items():
            if param_name in self._parameter_mapping:
                coin_name = self._parameter_mapping[param_name]
                parlist.append(coin_name)

                # Handle precision conversion for si_a and si_d
                if param_name in ['si_a', 'si_d']:
                    parvals.append(1 / (param_value ** 2))
                else:
                    parvals.append(param_value)
        return parlist, parvals

    def _configure_coin_object(self, nruns, n_ctx, max_cores):
        """Create and configure COIN inference object."""
        inf = coinp.COIN()

        # Set parameters
        parlist, parvals = self._setup_coin_parameters()
        for name, value in zip(parlist, parvals):
            setattr(inf, name, float(value))

        # Set configuration
        inf.store = self.store_fields
        inf.particles = self.particles
        inf.sigma_motor_noise = self.sigma_motor_noise
        inf.runs = nruns
        inf.max_contexts = n_ctx
        inf.max_cores = max_cores

        return inf

    def _compute_gaussian_likelihood(self, lambda_parts, y, mu_parts, sigma_parts):
        """Compute Gaussian likelihood for predictions."""
        return inference_utils.compute_gaussian_likelihood_weighted(lambda_parts, y, mu_parts, sigma_parts)

    def _compute_cumulative_distribution(self, lambda_parts, y, mu_parts, sigma_parts):
        """Compute cumulative distribution function values."""
        return inference_utils.compute_cumulative_distribution_weighted(lambda_parts, y, mu_parts, sigma_parts)

    def _compute_run_statistics(self, run_output, y_sequence, inf):
        """Compute all statistics for a single COIN run."""
        mu_parts = run_output["state_mean"]
        sigma_parts = np.sqrt(run_output["state_var"] + inf.sigma_sensory_noise**2)
        lambda_parts = run_output["predicted_probabilities"]

        # Predicted observations
        predictions = np.reshape(
            np.mean(np.sum(lambda_parts * mu_parts, axis=0), axis=0),
            (len(y_sequence),)
        )

        # Compute likelihoods
        p_y_parts = self._compute_gaussian_likelihood(lambda_parts, y_sequence, mu_parts, sigma_parts)
        log_probs = np.log(np.maximum(np.mean(p_y_parts, axis=0), np.finfo(float).eps))

        # Responsibilities
        responsibilities = np.reshape(
            np.mean(lambda_parts, axis=1),
            (lambda_parts.shape[0], len(y_sequence))
        )

        # Cumulative probabilities
        cumulative_probs = self._compute_cumulative_distribution(lambda_parts, y_sequence, mu_parts, sigma_parts)

        return {
            'predictions': predictions,
            'log_probs': log_probs,
            'responsibilities': responsibilities,
            'cumulative_probs': cumulative_probs
        }

    def _aggregate_run_results(self, all_run_stats, nruns):
        """Aggregate statistics across multiple runs."""
        return inference_utils.aggregate_run_results_coin(all_run_stats, nruns)

    def _process_single_sequence(self, y_sequence, inf, nruns):
        """Process one sequence through COIN inference."""
        inf.perturbations = y_sequence
        output = inf.simulate_coin()

        # Compute statistics for each run
        run_statistics = []
        for i in range(nruns):
            stats = self._compute_run_statistics(output["runs"][i], y_sequence, inf)
            run_statistics.append(stats)

        # Aggregate across runs
        return self._aggregate_run_results(run_statistics, nruns)

    def _process_batch_sequences(self, y, inf, nruns):
        """Process batch of sequences with progress tracking."""
        batch_results = {'predictions': [], 'log_probs': [], 'responsibilities': [], 'cumulative_probs': []}

        # Progress bar for batch processing
        batch_iterator = tqdm(range(y.shape[0]), desc="COIN inference") if y.shape[0] > 1 else range(y.shape[0])

        for b in batch_iterator:
            y_sequence = np.squeeze(y[b])
            sequence_results = self._process_single_sequence(y_sequence, inf, nruns)

            # Collect results
            for key in batch_results:
                batch_results[key].append(sequence_results[key])

        return batch_results

    def _format_outputs(self, batch_results):
        """Convert batch results to final output arrays."""
        z_coin = np.array(batch_results['predictions'])
        logp = np.array(batch_results['log_probs'])
        cump = np.array(batch_results['cumulative_probs'])
        lamb = np.array(batch_results['responsibilities'])

        return z_coin, logp, cump, lamb


    def estimate(self, y, nruns=1, n_ctx=10, max_cores=1):
        """Run COIN inference algorithm.

        Parameters: y (n_batches, n_trials), nruns, n_ctx, max_cores
        Returns: z_coin, logp, cump (n_batches, n_trials), lamb (n_batches, n_ctx, n_trials)
        """
        inf = self._configure_coin_object(nruns, n_ctx, max_cores)
        batch_results = self._process_batch_sequences(np.asarray(y), inf, nruns)
        return self._format_outputs(batch_results)


class LeakyAverageInference:
    """Leaky integrator inference engine with integrated configuration and optimization."""

    def __init__(self, generative_model, max_cores=None):
        self.generative_model = generative_model
        self.max_cores = max_cores
        self.best_tau = None

    def estimate(self, y, tau=None, force_sequential=False):
        """Run leaky integrator baseline.

        Parameters: y (n_batches, n_trials), tau (optional), force_sequential
        Returns: z_slid, logp, cump (n_batches, n_trials)
        """
        y = inference_utils.validate_input_basic(y)

        if tau is None:
            if self.best_tau is None:
                # Check if GenerativeModel has a cached best_t first
                if hasattr(self.generative_model, 'best_t'):
                    self.best_tau = self.generative_model.best_t
                else:
                    self._optimize_tau(n_trials=y.shape[1])
            tau = self.best_tau

        # Use utility function to determine processing mode
        if inference_utils.should_use_parallel(y.shape[0], force_sequential, self.max_cores):
            processor_func = lambda y_batch: self._compute_leaky_integrator(y_batch, tau)
            return inference_utils.process_parallel_batches(y, processor_func, self.max_cores)
        else:
            return self._compute_leaky_integrator(y, tau)

    def _optimize_tau(self, n_trials=5000, n_train_instances=500):
        """Find optimal tau for leaky integrator by minimizing prediction error."""
        X = self.generative_model.generate_batch(n_trials, n_train_instances)[0]

        def fn(tau):
            z_slid, _, _ = self._compute_leaky_integrator(X, tau)
            mse = ((z_slid - X)**2).mean()
            return mse

        optimRes = scipy.optimize.minimize_scalar(fn, bounds=(0, X.shape[1]),
                                                     method='bounded',
                                                     options={'xatol':1e-3})
        if optimRes.success:
            best_t = optimRes.x
        else:
            taus = np.arange(100, -1, -1)
            z_slid, _, _ = self._compute_leaky_integrator(X, tau=taus)
            best_t = taus[((z_slid[:, 0, :] - X)**2).mean((0,1)).argmin()]

        self.best_tau = best_t
        self.generative_model.best_t = best_t  # Sync with GenerativeModel for backward compatibility
        return best_t

    def _compute_leaky_integrator(self, y_batch, tau_val):
        """Core leaky integrator implementation."""
        if type(tau_val) != np.ndarray:
            tau_val = np.array([tau_val])

        weights = np.zeros((y_batch.shape[1], len(tau_val)))
        weights[:, tau_val>0] = np.exp(- np.einsum('T,m->Tm', np.arange(y_batch.shape[1], 0, -1), 1/tau_val[tau_val>0]))
        for tix in np.where(tau_val<=0)[0]:
            weights[:,tix] = np.eye(y_batch.shape[1])[-1]

        z_slid, s_slid = np.zeros((y_batch.shape[0],) + weights.shape), np.ones((y_batch.shape[0],) + weights.shape)

        for t in range(1, y_batch.shape[1]):
            w = weights[-t:,:] / weights[-t:, :].sum(0)
            z_slid[:, t, :] = np.einsum('bT,Tm->bm', y_batch[:,:t], w)
            s_slid[:, t, :] = np.sqrt(np.einsum('bTm,Tm->bm', (y_batch[:, :t, np.newaxis] - z_slid[:, :t,:])**2, w))

        logp, cump = inference_utils.compute_leaky_integrator_statistics(y_batch, z_slid, s_slid)

        if len(tau_val) == 1:
            z_slid, logp, cump = z_slid[..., 0], logp[..., 0], cump[..., 0]

        return z_slid, logp, cump



class GenerativeModel():
    """
    COIN Generative Model for context-dependent inference.

    Generates synthetic time series data using a hierarchical Bayesian framework
    where observations depend on latent contexts that evolve via Chinese Restaurant Process.

    Parameters
    ----------
    parset : str or dict
        Hyperparameter set name ('fitted', 'validation', 'training', etc.) or
        dict with 'name' and 'pars' fields containing hyperparameter values.

    Main Methods
    ------------
    Data Generation: generate_batch(), generate_session(), sample_*()
    Inference: estimate_coin(), estimate_leaky_average()
    Analysis: benchmark(), measure_ctx_transition_stats()
    """

    def __init__(self, parset):

        self.pool = ProcessingPool()

        if type(parset) is str:
            self.pars   = load_pars(parset)
            self.parset = parset
        else:
            self.pars   = parset['pars']
            self.parset = parset['name']

        # Initialize COIN inference engine
        self.coin_inference = COINInference(generative_params=self.pars)

        # Initialize leaky average inference engine
        self.leaky_average_inference = LeakyAverageInference(generative_model=self, max_cores=self.pars.get('max_cores'))

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

    def __del__(self):
        """Clean up ProcessingPool resources."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()

    # Auxiliary samplers
    def _sample_N_(self, mu, si, N=1):
        """Sample N values from normal distribution N(mu, si)."""

        return np.array(ss.norm.rvs(mu, si, N))

    def _sample_TN_(self, a, b, mu, si, N):
        """Sample N values from truncated normal distribution on [a,b]."""

        return np.array(ss.truncnorm.rvs((a-mu)/si, (b-mu)/si, mu, si, N))

    def _break_new_partition_(self, beta, gamma):
        """Stick-breaking step for GEM distribution - adds new partition to beta."""

        w = ss.beta.rvs(1, gamma) # Stick-breaking weight
        beta.append((1-w) * beta[-1])
        beta[-2] = w * beta[-2]
        return beta

    def _sample_customer_(self, beta, N, j, alpha, kappa=0):
        """Sample table assignment for customer j in Chinese Restaurant Process."""
        beta_w   = alpha * np.array(beta)
        sticky_w = 0 if kappa == 0 else kappa * np.eye(len(beta))[j]
        global_w = N[j, 0:len(beta)]
        return random.choices(range(len(beta)), beta_w + sticky_w + global_w, k=1)[0]

    # Main data sampler
    def generate_batch(self, n_trials, batch_size=1):
        """Generate batch of synthetic data.

        Parameters: n_trials (int), batch_size (int)
        Returns: y, q, c arrays with shape (n_batches, n_trials)
        """

        self.n_trials = n_trials

        seeds = np.random.randint(low=1, high=1024*16, size=(batch_size)).cumsum()
        
        if self.max_cores is None or self.max_cores > 1:
            pool = ProcessingPool(nodes=self.max_cores)
            res = pool.map(self.generate_session, seeds)
        else:
            res = [self.generate_session(seed) for seed in seeds]

        y = np.concatenate([r[0] for r in res], axis=0)
        q = np.concatenate([r[1] for r in res], axis=0)
        c = np.concatenate([r[2] for r in res], axis=0)

        self.n_trials = None

        return(y, q, c)

    def generate_context_batch(self, n_trials, batch_size=1):
        """Generate batch of context sequences.

        Returns dict with 'C' field containing contexts (n_batches, n_trials).
        May include 'pi' and 'beta' fields for ground truth statistics.
        """

        self.n_trials = n_trials
        seeds = np.random.randint(low=1, high=1024*16, size=(batch_size)).cumsum()

        if self.max_cores is None or self.max_cores > 1:
            # res = self.pool.map(self.sample_contexts, seeds)
            pool = ProcessingPool(nodes=self.max_cores)
            res = pool.map(self.sample_contexts, seeds)
        else:
            res = [self.sample_contexts(seed) for seed in seeds]


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
        """Generate single data sequence.

        Parameters: seed (int, optional), n_trials (int)
        Returns: y, q, c arrays with shape (1, n_trials)
        """

        if n_trials is not None:
            self.n_trials = n_trials

        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)

        c = np.zeros((1, self.n_trials), int)
        q = np.zeros((1, self.n_trials), int)
        y = np.zeros((1, self.n_trials))

        contex_sample_results = self.sample_contexts()
        if type(contex_sample_results) is list:
            contexts = contex_sample_results
        elif type(contex_sample_results) is tuple:
            contexts = contex_sample_results[0]
        
        states   = self.sample_states(contexts)
        y[0, :]  = self.sample_observations(contexts, states)
        q[0, :]  = self.sample_cues(contexts)
        c[0, :]  = copy.deepcopy(contexts)

        if seed is not None:
            np.random.set_state(random_state)

        return(y, q, c)

    def sample_observations(self, contexts, states):
        """Generate observations y_t from contexts and states.

        Returns: observations (1, n_trials)
        """

        y = np.zeros((1, len(contexts)))
        v = self._sample_N_(0, self.si_r, len(contexts))

        for t, c in enumerate(contexts):
            y[0, t] = states[c][t] + v[t]

        return y

    def sample_states(self, contexts, return_pars=False):
        """Generate latent states for each context using AR(1) dynamics.

        Returns: states dict, optionally (a, d) if return_pars=True
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

    def sample_contexts(self, seed=None, n_trials=None):
        """Sample context sequence via Markov chain simulation.

        Returns: list of context integers
        """

        if seed is not None:
            random_state_contexts = np.random.get_state()
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

        if seed is not None:
            np.random.set_state(random_state_contexts)

        return contexts

    def sample_cues(self, contexts):
        """Sample cue sequence given contexts. Cue emission is untested.

        Returns: list of cue integers
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
        """Run COIN inference algorithm.

        Parameters: y (n_batches, n_trials), nruns, n_ctx, max_cores
        Returns: z_coin, logp, cump (n_batches, n_trials), lamb (n_batches, n_ctx, n_trials)
        """
        return self.coin_inference.estimate(y, nruns, n_ctx, max_cores)

    # Baselines and heuristic estimations
    def empirical_expected_beta(self, n_contexts=11, n_trials=1000, n_samples=1000):
        """Compute empirical expected global distribution of contexts."""
        c_series  = self.generate_context_batch(n_trials, n_samples)['C']

        # Empirical distribution of contexts in each sample
        empirical = [[(c == ctx).mean() for ctx in range(n_contexts)] for c in c_series]

        # Single sample of uniform distribution for contexts with p < 1/n_samples
        uniform   = [1/n_contexts for ctx in range(n_contexts)]

        # Global empirical distribution
        global_distribution = np.array(empirical + [uniform]).mean(0)

        return global_distribution

    def estimate_leaky_average(self, y, tau=None, force_sequential=False):
        """Run leaky integrator baseline.

        Parameters: y (n_batches, n_trials), tau (optional), force_sequential
        Returns: z_slid, logp, cump (n_batches, n_trials)
        """
        return self.leaky_average_inference.estimate(y, tau, force_sequential)

    def fit_best_tau(self, n_trials=5000, n_train_instances=500):
        """Find optimal tau for leaky integrator by minimizing prediction error."""
        best_t = self.leaky_average_inference._optimize_tau(n_trials, n_train_instances)
        self.best_t = best_t  # Maintain backward compatibility
        return best_t

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
            z_slid, p_slid, cump_slid = self.estimate_leaky_average(X, tau)

            LI_mse = ((z_slid - X)**2).mean(1)
            LI_kol = inference_utils.compute_kolmogorov_statistic(cump_slid, n_trials)
            LI_ce  = p_slid.mean(1) # Observations cross entropy
            LI_ct_ac = (C == 0).mean(1) # LI assumed to predict always context 0 # Context identification accuracy
            LI_ct_p  = (C == 0).mean(1) # Probability of the actual context on the posterior of the prediction
            # p_ctx not defined in LI --> we predict instead the empirical global distribution across c
            empirical_beta = self.empirical_expected_beta(n_trials=n_trials, n_samples=100*n_instances) 
            LI_ct_ce = np.zeros(C.shape[0])
            for b in range(C.shape[0]):
                for ctx in range(len(empirical_beta)):
                    LI_ct_ce[b] += np.log(empirical_beta[ctx]) * (C[b, :] == ctx).mean()

            print(f'Estimating coin...', flush=True)

            z_coin, ll_coin, cump_coin, lamb = self.estimate_coin(X, n_ctx=64)
            loglamb = np.log(lamb + np.exp(minloglik))

            coin_mse = ((z_coin - X)**2).mean(1)
            coin_kol = inference_utils.compute_kolmogorov_statistic(cump_coin, cump_coin.shape[1])
            coin_ce  = ll_coin.mean(1)

            predicted_contexts = np.argmax(lamb, axis=1) # Predicted context
            coin_ct_ac = (predicted_contexts == C).mean(1) # Context identification accuracy
            coin_ct_p, coin_ct_ce = inference_utils.compute_context_probability_statistics(lamb, C, minloglik)


            perf = {'LI' :  {'mse': {}, 'kol': {}, 'ce': {}, 'ct_ac': {}, 'ct_p': {}, 'ct_ce': {}}, 
                    'coin': {'mse': {}, 'kol': {}, 'ce': {}, 'ct_ac': {}, 'ct_p': {}, 'ct_ce': {}}}

            # Observations MSE
            perf['LI']['mse']     = inference_utils.compute_mean_sem(LI_mse, n_instances)
            perf['coin']['mse']   = inference_utils.compute_mean_sem(coin_mse, n_instances)
            # Observations calibration
            perf['LI']['kol']     = inference_utils.compute_mean_sem(LI_kol, n_instances)
            perf['coin']['kol']   = inference_utils.compute_mean_sem(coin_kol, n_instances)
            # Observations cross-entropy
            perf['LI']['ce']      = inference_utils.compute_mean_sem(LI_ce, n_instances)
            perf['coin']['ce']    = inference_utils.compute_mean_sem(coin_ce, n_instances)
            # Context identification accuracy
            perf['LI']['ct_ac']   = inference_utils.compute_mean_sem(LI_ct_ac, n_instances)
            perf['coin']['ct_ac'] = inference_utils.compute_mean_sem(coin_ct_ac, n_instances)
            # Probability of the actual context on the posterior of the prediction
            perf['LI']['ct_p']    = inference_utils.compute_mean_sem(LI_ct_p, n_instances)
            perf['coin']['ct_p']  = inference_utils.compute_mean_sem(coin_ct_p, n_instances)
            # Context cross-entropy
            perf['LI']['ct_ce']   = inference_utils.compute_mean_sem(LI_ct_ce, n_instances)
            perf['coin']['ct_ce'] = inference_utils.compute_mean_sem(coin_ct_ce, n_instances)


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
        # Note: Some parameters use default COIN implementation values (marked with *)
        pars = {}
        if parset == 'fitted': # Average of evoked+spontaneous across subjects = implementation
            pars['alpha_t'] = 9.0
            pars['gamma_t'] = 0.1    # *
            pars['rho_t']   = 0.25
            pars['alpha_q'] = 25.0   # *
            pars['gamma_q'] = 0.1    # *
            pars['mu_a']    = 0.9425
            pars['si_a']    = 0.0012
            pars['si_d']    = 0.0008
            pars['si_q']    = 0.0089
            pars['si_r']    = 0.03   # *
        elif parset == 'validation':
            pars['alpha_t'] = 10.0
            pars['gamma_t'] = 0.1    # *
            pars['rho_t']   = 0.9
            pars['alpha_q'] = 10.
            pars['gamma_q'] = 0.1    # *
            pars['mu_a']    = 0.9
            pars['si_a']    = 0.1
            pars['si_d']    = 0.1
            pars['si_q']    = 0.1
            pars['si_r']    = 0.03   # *
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

