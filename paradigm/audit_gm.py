import numpy as np
import time

import sys
import os
import scipy.stats as ss


import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm

from matplotlib.ticker import MaxNLocator

import random


class HierarchicalGenerativeModel():

    def __init__(self, params):

        self.params = params

        self.N_batch = params["N_batch"]
        self.N_blocks = params["N_blocks"]
        self.N_tones = params["N_tones"]
        self.N_rules = params["N_rules"]
        self.rules_dpos_set = params["rules_dpos_set"]
        self.tones_values = params["tones_values"]
        self.mu_tau = params["mu_tau"]
        self.si_tau = params["si_tau"]
        self.si_lim = params["si_lim"]
        self.mu_rho_rules = params["mu_rho_rules"]
        self.si_rho_rules = params["si_rho_rules"]
        self.mu_rho_timbres = params["mu_rho_timbres"]
        self.si_rho_timbres = params["si_rho_timbres"]        
        self.si_q = params["si_q"]
        self.si_r = params["si_r"]


        pass

    # Auxiliary samplers from goin.coin.GenerativeModel
    def _sample_N_(self, mu, si, size=1):
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

    def _sample_TN_(self, a, b, mu, si, size=1):
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
    

    def sample_pair(self, tones): 
        pair = np.random.choice(tones, size=(2,), replace=False)     
        # y_std = pair[0]
        # y_dvt = pair[1]
        return pair

    def sample_next_markov_state(self, current_state, states_values, states_trans_matrix):
        return np.random.choice(states_values, p=states_trans_matrix[current_state])
    

    def sample_pi(self, N, mu_rho, si_rho):
        """A transition matrix with a sticky diagonal

        Parameters
        ----------
        mu_rho : _type_
            _description_
        si_rho : _type_
            _description_

        Returns
        -------
        np.array (N, N)
            Transition matrix
        """

        # Sample parameters
        rho = self._sample_TN_(0, 1, mu_rho, si_rho).item()
        eps = [np.random.uniform() for n in range(N)]
        

        # Delta has a zero diagonal and the rest of the elements of a row (for a rule) are partitions from 1 using the corresponding eps[row] (parameter for that rule), controlling for the sum to be 1
        delta = np.array([[0 if i == j else eps[i]*(1-eps[i])**j if (j<i and j<N-2) else eps[i]*(1-eps[i])**(j-1) if i<j<N-1 else 1 - sum([eps[i]*(1-eps[i])**k for k in range(N-2)]) for j in range(N)] for i in range(N)])
        
        # Transition matrix
        pi = rho * np.eye(N) + (1-rho) * delta
        return pi
    
    
    def sample_rules(self, N_blocks, N_rules, mu_rho_rules, si_rho_rules):
        """Sample rules for a run consisting in a sequence of blocks of tones (each sequence being associated with one rule).
        Rules evolve in a Markov chain manner.

        Parameters
        ----------
        N_blocks : int
            Number of blocks of tones
        N_rules : int
            Number of rules
        mu_rho_rules : float
            _description_
        si_rho_rules : float
            _description_

        Returns
        -------
        np.array (N_blocks,)
            Sequence of blocks' rules
        """

        # Sample rules transition matrix is sampled initially from a parametric distribution
        pi_rules = self.sample_pi(N_rules, mu_rho_rules, si_rho_rules)
        # pi_rules_0   = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]) # A very sticky transition matrix

        # Rules sequence
        rules = np.zeros(N_blocks, dtype=np.int64)

        # Initilize rule (assign to 0, randomly, or from the distribution from which the transition probas also come from)
        rules[0] = 0
        # rules[0] = np.random.choice(N_rules)

        for s in range(1, N_blocks):
            # Markov chain
            # rules[s] = self.sample_next_markov_state(rules[s-1], range(N_rules), pi_rules)
            rules[s] = np.random.choice(range(N_rules), p=pi_rules[rules[s-1]])

        return rules


    def sample_timbres(self, rules_seq, N_timbres, mu_rho_timbres, si_rho_timbres):
        """_summary_

        Parameters
        ----------
        rules_seq : np.array
            _description_
        N_timbres : int
            _description_
        mu_rho_timbres : float
            _description_
        si_rho_timbres : float
            _description_

        Returns
        -------
        list
            _description_
        """

        # Sample timbres transition (emission from rule) matrix
        pi_timbre = self.sample_pi(N_timbres, mu_rho_timbres, si_rho_timbres)

        timbres = np.array([np.random.choice(range(N_timbres), p=pi_timbre[seq]) for seq in rules_seq])
        
        return timbres


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

        pi_tilde = self._sample_GEM_(alpha, threshold=1E-8)
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

    def sample_pi_t(self, alpha, beta, rho_rules):
        """Samples a finite approximation of the ground truth transition probability matrix

        Parameters
        ----------
        alpha  : float
            concentration parameter controls the dispersion of the rows with respect to beta 
        beta : np.array (a discrete probability distribution)
            global probabilities
        rho_rules : float 
            normalised self-transition bias; 0 < rho_rules < 1
        
        Returns
        -------
        np.array
           transition probability matrix; row n is the transition distribution from context n
        """

        pi = np.zeros((len(beta), len(beta)))
        for j in range(len(beta)):
            delta = np.eye(len(beta))[j]
            pi[j, :] = self._sample_DP_(alpha / (1-rho_rules), (1-rho_rules) * beta + rho_rules * delta)

        return pi

    def sample_rules_GEM(self, seed=None, n_trials=None):
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
        rules : np.array 
            sequence of sampled rules; dim 0 runs across time points, dim 1 is set to one for
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
        rules = random.choices(range(len(beta_t)), beta_t)
        #contexts = [0]

        n_ctx = len(beta_t)
        for t in range(1, self.n_trials):
            rules += random.choices(range(n_ctx), pi_t[rules[t-1], :])

        self.beta_t = beta_t
        self.pi_t   = pi_t
        self.n_ctx  = n_ctx

        return(rules, beta_t, pi_t)

    
    
    def sample_dpos(self, rules, rules_dpos_set):
        """Sample positions of the deviant tones for each block of tones

        Parameters
        ----------
        rules : np.array
            Sequence of blocks rules_
        rules_dpos_set : np,array (N_rules, 3)
            Mapping of the 3 indexes of possible deviant positions for each of the rules

        Returns
        -------
        list
            List of each block of tones' deviant position index
        """

        #Given the trial's rule stored in rules and the possible positions for deviant associated with it, sample the position where the deviant will be located in this trial
        return [np.random.choice(rules_dpos_set[rule]) for rule in rules]


    
    def get_contexts(self, dpos, N_blocks, N_tones):
        contexts = np.zeros((N_blocks, N_tones), dtype=np.int64)
        for i, pos in enumerate(dpos):
            contexts[i, pos] = 1
        return contexts

    
    def sample_states(self, contexts, return_pars=False):
        """Generates a single data sequence y_t given a sequence of contexts c_t a sequence of 
        states x_t^c containing the states of standard and deviant

        Here contexts is the sequence of tone-by-tone boolean value stnding for {std, dvt} that has been hierararchically defined prior to the call of sample_states
        

        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_blocks, N_tones)
        return_pars: bool
            also returns the retention and drift parameters for each state


        Returns
        -------
        states : dict
            dictionary encoding the latent state values (one-dimensional np.array) for each 
            context c (keys).
        tau: time constants parameters for each context (only if return_pars set to True)
        d: steady state parameters for each context (only if return_pars set to True)

        """

        # # Note that retention and drift are sampled in every call
        # a = self._sample_TN_(0, 1, self.mu_a, self.si_a, (np.max(contexts)+1, contexts.shape[0])) # TODO: check if okay to replace with len(np.unique(contexts))
        # d = self._sample_N_(0, self.si_d, (np.max(contexts)+1, contexts.shape[0]))


        # Sample params for each block
        tau, lim = np.zeros((2,self.N_blocks)), np.zeros((2,self.N_blocks)) # 2 as for len({std, dvt})
        for b in range(self.N_blocks):
            # Sample one pair of std/dvt lim values for each block
            mu_lim = self.sample_pair(self.tones_values)

            for c in np.unique(contexts): # np.unique(contexts)=range(2)
                # Sample dynamics params for each context (std and dvt)
                tau[c,b] = self._sample_TN_(1, 50, self.mu_tau, self.si_tau).item() # A high boundary
                lim[c,b] = self._sample_N_(mu_lim[c], self.si_lim).item() # TODO: check values

        # tau = self._sample_TN_(0, 1, self.mu_tau, self.si_tau, (np.max(contexts)+1, contexts.shape[0])) # TODO: check if okay to replace with len(np.unique(contexts))
        # lim = self._sample_N_(self.mu_lim, self.si_lim, (np.max(contexts)+1, contexts.shape[0]))

        states = dict([(c, np.zeros(contexts.shape)) for c in np.unique(contexts)])

        for c in np.unique(contexts): # np.unique(contexts)=range(2)

            # Initialize with a sample from distribution of mean and std the LGD stationary values
            # states[c][:,0] = self._sample_N_(d[c]/(1-a[c]), self.si_q/((1-a[c]**2)**.5), (contexts.shape[0], 1))
            states[c][:,0] = self._sample_N_(lim[c,:], self.si_q*tau[c,:]/((2*tau[c,:]-1)**.5), (contexts.shape[0],))

            for b in range(self.N_blocks): 

                # Sample noise
                w = self._sample_N_(0, self.si_q, contexts.shape)
            
            
                # Here the states exist independently of the contexts

                for t in range(1, contexts.shape[1]):
                    # states[c][:,t] = a[c] * states[c][:,t-1] + d[c] + w[:,t-1]
                    states[c][b,t] = states[c][b,t-1] + 1 / tau[c,b] * (lim[c,b] - states[c][b,t-1]) + w[b,t-1]   


        if return_pars:
            # return states, a, d
            return states, tau, lim
        else:
            return states
        

    def sample_observations(self, contexts, states):
        """Generates a single data sequence y_t given a sequence of contexts c_t and a sequence of 
        states x_t^c

        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_blocks, N_tones)
        states : dict
            dictionary encoding the latent state values (one-dimensional np.array) for each 
            context c (keys).

        Returns
        -------
        y  : np.array
            2-dimensional sequence of observations of size (N_blocks, N_tones)
        """

        obs = np.zeros(contexts.shape)
        v = self._sample_N_(0, self.si_r, contexts.shape)

        for (s,t), c in np.ndenumerate(contexts):
            # Noisy observation of one of the two states (std or dvt), as imposed by the current context c
            obs[s,t] = states[c][s,t] + v[s,t]

        return obs

        
    
    def generate_run(self):

        # Sample rules
        rules = self.sample_rules(self.N_blocks, self.N_rules, self.mu_rho_rules, self.si_rho_rules)
        rules_long = np.tile(rules[:,np.newaxis], (1,self.N_tones)) # Store latent rules in a per-tone array # This is equivalent to matlab's repmat

        # Sample timbres (here we consider that there are as many different timbres as there are different rules -- self.N_rules)
        timbres = self.sample_timbres(rules, self.N_rules, self.mu_rho_timbres, self.si_rho_timbres)

        # Sample deviant position
        dpos = self.sample_dpos(rules, self.rules_dpos_set)

        # Get contexts
        contexts = self.get_contexts(dpos, self.N_blocks, self.N_tones)

        # Sample states and observations
        states  = self.sample_states(contexts)
        obs     = self.sample_observations(contexts, states)


        # Flatten rules_long, contexts, (states, ) timbres and obs
        rules_long  = rules_long.flatten()
        contexts    = contexts.flatten()
        states      = dict([(key, states[key].flatten()) for key in states.keys()])
        obs         = obs.flatten()

        return rules, rules_long, dpos, timbres, contexts, states, obs 
    


    def generate_batch(self, N_batch):
        # Store latent rules and timbres, states and observations
        

        for batch in range(N_batch):
            # Generate a batch of N_blocks sequences, sampling parameters and generating the paradigm's observations
            rules, timbres, states, obs = self.generate_run()



    def plot_contexts_states_obs(self, Cs, ys, y_stds, y_dvts, T):
        """For a non-hierarchical situation (only contexts std/dvt, no rules)

        Parameters
        ----------
        Cs : _type_
            sequence of contexts
        ys : _type_
            observations
        y_stds : _type_
            states of std
        y_dvts : _type_
            states of dvt
        """

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(y_stds, label='y_std', color='green', linestyle='dotted', linewidth=2)
        ax1.plot(y_dvts, label='y_dvt', color='blue', linestyle='dotted', linewidth=2)
        ax1.plot(ys, label='y', color='red', linestyle='dashed', linewidth=2)
        ax1.set_ylabel('y')

        ax2 = ax1.twinx()
        ax2.plot(range(T), Cs, 'o', color='black', label='context')
        ax2.set_ylabel('context')
        ax2.set_yticks(ticks=[0,1], labels=['std', 'dvt'])


        fig.legend()

        fig.tight_layout()
        plt.show()


    def plot_contexts_rules_states_obs(self, y_stds, y_dvts, ys, Cs, rules, dpos):
        """For the hierachical evolution of rules and contexts (NOTE: timbres not included in this viz atm)

        Parameters
        ----------
        y_stds : _type_
            _description_
        y_dvts : _type_
            _description_
        ys : _type_
            _description_
        Cs : _type_
            _description_
        rules : _type_
            _description_
        """
        
        # Visualize tone frequencies
        fig, ax1 = plt.subplots(figsize=(20, 6))
        ax1.plot(y_stds, label='y_std', color='green', marker='o', markersize=4, linestyle='dotted', linewidth=2, alpha=0.5)
        ax1.plot(y_dvts, label='y_dvt', color='blue', marker='o', markersize=4, linestyle='dotted', linewidth=2, alpha=0.5)
        ax1.plot(ys, label='y', color='k', marker='o', markersize=4, linestyle='dashed', linewidth=2)
        ax1.set_ylabel('y')

        ax2 = ax1.twinx()
        ax2.plot(Cs, 'o', color='black', label='context', markersize=2)
        ax2.set_ylabel('context')
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_yticks(ticks=[0,1], labels=['std', 'dvt'])

        rules_cmap = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:orange'} 
        for i, rule in enumerate(rules):
            plt.axvspan(i*self.N_tones, i*self.N_tones + self.N_tones, facecolor=rules_cmap[rule], alpha=0.25)
            
        for i in range(self.N_blocks):
            plt.axvline(i*self.N_tones, color='tab:gray', linewidth=0.9)
            ax2.text(x=i*self.N_tones+0.35*self.N_tones, y=0.95, s=f'rule {rules[i]}', color=rules_cmap[rules[i]])
            ax2.text(x=i*self.N_tones+0.35*self.N_tones, y=0.85, s=f'dvt {dpos[i]}', color=rules_cmap[rules[i]])

        fig.legend(bbox_to_anchor=(1.1, 1)) #, loc='upper left'bbox_to_anchor=(1.05, 1), frameon=False)
        # fig.tight_layout(rect=[0, 0, 0.85, 1])
        # fig.legend()
        plt.tight_layout()
        plt.show()


    def plot_rules_dpos(self, rules, dpos):

        # Visualize hierarchical information: dvt pos and rule

        rules_cmap = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:orange'} 

        fig, ax = plt.subplots(figsize=(20, 6))
        for i, y in enumerate(dpos):
            ax.vlines(x=i, ymin=0, ymax=y, color='tab:gray', linewidth=0.9, zorder=1, alpha=0.5)   
        ax.scatter(range(len(dpos)), dpos, c=[rules_cmap[rule] for rule in rules], zorder=2)
        ax.set_ylabel('dvt pos')
        ax.set_xlabel('trial')
        ax.set_ylim(1, 8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(range(len(dpos)))
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in rules_cmap.values()]
        labels = rules_cmap.keys()
        ax.legend(handles, labels, title='rule')
        fig.tight_layout()
        plt.show()




if __name__ == '__main__':

    config = {
        "N_batch": 1,
        "N_blocks": 20,
        "N_tones": 8,
        "N_rules": 3,
        "rules_dpos_set": np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]]),
        "tones_values": [1455, 1500, 1600],
        "mu_tau": 4,
        "si_tau": 1,
        "si_lim": 5,
        "mu_rho_rules": 0.9,
        "si_rho_rules": 0.05,
        "mu_rho_timbres": 0.8,
        "si_rho_timbres": 0.05,
        "si_q": 2,
        "si_r": 2
    }
    
    gm = HierarchicalGenerativeModel(config)

    rules, rules_long, dpos, timbres, contexts, states, obs = gm.generate_run()

    gm.plot_contexts_rules_states_obs(states[0][0:gm.N_tones], states[1], obs, contexts, rules, dpos)
    gm.plot_rules_dpos(rules, dpos)
    gm.plot_contexts_states_obs(contexts[0:gm.N_tones], obs[0:gm.N_tones], states[0][0:gm.N_tones], states[1][0:gm.N_tones], gm.N_tones)

    pass
    