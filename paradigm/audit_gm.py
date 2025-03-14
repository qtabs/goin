import numpy as np
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from goin import coin as coin
from goin.opt_coin.test_opt_coin_inf import compute_logpc
import scipy.stats as ss


import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm

from matplotlib.ticker import MaxNLocator


class HierarchicalGenerativeModel():

    def __init__(self, params):
        # Rules and timbres transition probabilities distribution parameters



        pass

    # Auxiliary samplers from goin.coin.GenerativeModel
    def _sample_N_(mu, si, size=1):
        """Samples from a normal distribution

        Parameters
        ----------
        mu : float
            Mean of the normal distribution
        si : float
            Standard deviation of the normal distribution
        size  : int or tuple of int (optional)
            Size of samples

        Returns
        -------
        np.array
            samples
        """

        return np.array(ss.norm.rvs(mu, si, size))

    def _sample_TN_(a, b, mu, si, size):
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
        size  : int or tuple of int (optional)
            Size of samples

        Returns
        -------
        np.array
            samples
        """

        return np.array(ss.truncnorm.rvs((a-mu)/si, (b-mu)/si, mu, si, size))

    def sample_rules(self):
        pass

    def sample_timbres(self):
        # Equivalent to cue
        pass

    def samle_dpos(self):
        pass
    
    
    def get_contexts(self):
        pass

    def sample_states(self, contexts, return_pars=False):
        """Generates a single data sequence y_t given a sequence of contexts c_t a sequence of 
        states x_t^c containing the states of standard and deviant

        Here contexts is the sequence of tone-by-tone boolean value stnding for {std, dvt} that has been hierararchically defined prior to the call of sample_states
        

        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_seq, L_seq)
        return_pars: bool
            also returns the retention and drift parameters for each state


        Returns
        -------
        states : dict
            dictionary encoding the latent state values (one-dimensional np.array) for each 
            context c (keys).
        a: retention parameters for each context (only if return_pars set to True)
        d: drift parameters for each context (only if return_pars set to True)

        """

        # Note that retention and drift are sampled in every call
        a = self._sample_TN_(0, 1, self.mu_a, self.si_a, (np.max(contexts)+1, contexts.shape[0])) # TODO: check if okay to replace with len(set(contexts))
        d = self._sample_N_(0, self.si_d, (np.max(contexts)+1, contexts.shape[0]))

        states = dict([(c, np.zeros(contexts.shape)) for c in set(contexts)])
        for c in set(contexts):
            # Initialize with a sample from distribution of mean and std the LGD stationary values
            states[c][:,0] = self._sample_N_(d[c]/(1-a[c]), self.si_q/((1-a[c]**2)**.5), (contexts.shape[0], 1))
            w = self._sample_N_(0, self.si_q, contexts.shape)
            # Here the states exist independently of the contexts
            for t in range(1, contexts.shape[1]):
                states[c][:,t] = a[c] * states[c][:,t-1] + d[c] + w[:,t-1]

        if return_pars:
            return states, a, d
        else:
            return states

    def sample_observations(self, contexts, states):
        """Generates a single data sequence y_t given a sequence of contexts c_t and a sequence of 
        states x_t^c

        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_seq, L_seq)
        states : dict
            dictionary encoding the latent state values (one-dimensional np.array) for each 
            context c (keys).

        Returns
        -------
        y  : np.array
            2-dimensional sequence of observations of size (N_seq, L_seq)
        """

        y = np.zeros(contexts.shape)
        v = self._sample_N_(0, self.si_r, contexts.shape)

        for (s,t), c in np.ndenumerate(contexts):
            y[s,t] = states[c][s,t] + v[s,t]

    
    


    def generate_variables(self):

        # Sample rules
        rules = self.sample_rules()

        # Sample cues
        timbres = self.sample_timbres(rules)

        # Sample deviant position

        # Get contexts


        # Sample states and observations
        states  = self.sample_states(rules)
        obs     = self.sample_observations(rules, states)


        return rules, timbres, states, obs