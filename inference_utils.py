"""Utility functions for COIN and LeakyAverage inference engines."""

import numpy as np
import scipy.stats
import scipy.special
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool


def compute_gaussian_likelihood_weighted(lambda_parts, y, mu_parts, sigma_parts):
    """Compute weighted Gaussian likelihood for COIN inference."""
    y_expanded = np.tile(y, (mu_parts.shape[0], mu_parts.shape[1], 1))

    normalization = lambda_parts / (np.sqrt(2 * np.pi) * sigma_parts)
    exponent = np.exp(-(y_expanded - mu_parts)**2 / (2 * sigma_parts**2))
    weighted_likelihood = normalization * exponent

    return np.sum(weighted_likelihood, axis=0)


def compute_cumulative_distribution_weighted(lambda_parts, y, mu_parts, sigma_parts):
    """Compute weighted cumulative distribution for COIN inference."""
    y_expanded = np.tile(y, (mu_parts.shape[0], mu_parts.shape[1], 1))
    cdf_values = scipy.stats.norm.cdf(y_expanded, mu_parts, sigma_parts)
    weighted_cdf = np.sum(lambda_parts * cdf_values, axis=0)
    mean_cdf = np.mean(weighted_cdf, axis=0)

    return np.reshape(mean_cdf, (y.shape[0],))


def compute_leaky_integrator_statistics(y_batch, z_slid, s_slid):
    """Compute log-probabilities and cumulative probabilities for leaky integrator."""
    logp = scipy.stats.norm.logpdf(y_batch[:, :, np.newaxis], z_slid, s_slid)
    cump = scipy.stats.norm.cdf(y_batch[:, :, np.newaxis], z_slid, s_slid)
    return logp, cump


def aggregate_run_results_coin(all_run_stats, nruns):
    """Aggregate COIN statistics across multiple runs using log-sum-exp."""
    mu_runs = np.column_stack([stats['predictions'] for stats in all_run_stats])
    logp_runs = np.column_stack([stats['log_probs'] for stats in all_run_stats])
    lamb_runs = np.stack([stats['responsibilities'] for stats in all_run_stats], axis=2)
    cump_runs = [stats['cumulative_probs'] for stats in all_run_stats]

    predictions = np.mean(mu_runs, axis=1)
    log_probs = scipy.special.logsumexp(logp_runs, axis=1) - np.log(nruns)
    responsibilities = np.mean(lamb_runs, axis=2)
    cumulative_probs = np.concatenate(cump_runs)

    results = {
        'predictions': predictions,
        'log_probs': log_probs,
        'responsibilities': responsibilities,
        'cumulative_probs': cumulative_probs
    }
    return results


def process_parallel_batches(y, processor_func, max_cores):
    """Process batches in parallel and aggregate results."""
    args_list = [(y[i:i+1], ) for i in range(y.shape[0])]

    pool = ProcessingPool(nodes=max_cores)
    results = pool.map(lambda args: processor_func(*args), args_list)

    # Aggregate results - handle leaky integrator format (z, logp, cump)
    z_slid = np.array([r[0][0, :] if r[0].ndim > 1 else r[0] for r in results])
    logp = np.array([r[1][0, :] if r[1].ndim > 1 else r[1] for r in results])
    cump = np.array([r[2][0, :] if r[2].ndim > 1 else r[2] for r in results])

    return z_slid, logp, cump


def create_batch_progress_iterator(n_batches, description):
    """Create progress iterator for batch processing."""
    return tqdm(range(n_batches), desc=description) if n_batches > 1 else range(n_batches)


def should_use_parallel(n_batches, force_sequential, max_cores):
    """Determine if parallel processing should be used."""
    has_multiple_batches = n_batches > 1
    parallel_allowed = not force_sequential
    cores_available = max_cores is None or max_cores > 1

    return has_multiple_batches and parallel_allowed and cores_available


def validate_input_basic(y):
    """Basic input validation for research code."""
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"Input must be 2D (n_batches, n_trials), got shape {y.shape}")
    return y