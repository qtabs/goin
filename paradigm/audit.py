import numpy as np
import time

import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from goin import coin as coin
# from goin.opt_coin.test_opt_coin_inf import compute_logpc
import scipy.stats as ss


import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm

from matplotlib.ticker import MaxNLocator

# Auxiliary samplers from goin.coin.GenerativeModel
def _sample_N_(mu, si, N=1):
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

def _sample_TN_(a, b, mu, si, N):
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

# Local functions

def dirichlet_func(x, val):
    if x==val:
        y=1
    else:
        y=0
    return y


def sample_obs(y_std, y_dvt, ctx, sigma_sampling_noise):
    freq = y_std * dirichlet_func(ctx, 0) + y_dvt * dirichlet_func(ctx, 1) + sigma_sampling_noise * np.random.randn()
    return freq


def sample_markov_state(current_state, states_values, states_trans_matrix):
    return np.random.choice(states_values, p=states_trans_matrix[current_state])


def sample_uniform_pair(y_values): 
    pair = np.random.choice(y_values, size=(2,), replace=False)     
    y_std = pair[0]
    y_dvt = pair[1]
    return y_std, y_dvt


def sample_dvt_pos(rule, rules_dvt_pos):
    return np.random.choice(rules_dvt_pos[rule])


def sample_timbre(timbre, timbre_set, timbre_rule_assoc):
    pass
    # np.random.choice(context_states, p=context_trans)



def generate_observations(mu_a = 0.85, si_a = 0.05, si_d = 5, sigma_freq_noise = 5, T_max = 512, y_std = 1550, y_dvt = 1500, sigma_sampling_noise = 5, C0 = 0, C_trans = np.array([[0.9, 0.1], [0.9, 0.1]]), contexts_set=[0,1]):

    # Std and dvt distributions dynamics param
    # time
    # Stationary freq values
    # Sampling noise
    # Context states 
    # State transition dynamics
    
    
    # Contexts sequence filled with initial context state
    Cs = np.zeros(T_max, dtype=np.int64)
    ctx = C0

    # Store std and dvt dynamics for visualization
    y_stds, y_dvts = np.zeros(T_max), np.zeros(T_max)

    # Observations sequence
    ys = np.zeros(T_max)
    ys[0] = sample_obs(y_stds[0], y_dvts[0], Cs[0], sigma_sampling_noise)

    a_std = _sample_TN_(0, 1, mu_a, si_a, 1).item()
    d_std = _sample_N_(0, si_d, 1)
    a_dvt = _sample_TN_(0, 1, mu_a, si_a, 1).item()
    d_dvt = _sample_N_(0, si_d, 1)


    for t in range(T_max):
        # Linear gaussian dynamics
        y_std = y_std * a_std + d_std + sigma_freq_noise * np.random.randn() # eps is some random gaussian noise
        y_dvt = y_dvt * a_dvt + d_dvt + sigma_freq_noise * np.random.randn() # the std and dvt could have different noise variance, TBD later

        y_stds[t] = y_std.item()
        y_dvts[t] = y_dvt.item()

        # First-order Markovian context dynamics
        ctx = sample_markov_state(ctx, contexts_set, C_trans)
        Cs[t] = ctx

        # Sample frequence based on context at stake
        ys[t] = sample_obs(y_stds[t], y_dvts[t], Cs[t], sigma_sampling_noise)


    return Cs, ys, y_stds, y_dvts


def plot_contexts_states_obs(Cs, ys, y_stds, y_dvts):

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



def plot_contexts_inference(Cs, c_hat, logp_c):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(range(T), Cs, 'bo', label='C', linewidth=2)
    ax1.plot(range(T), c_hat.squeeze(), 'ro', label='C_hat', linewidth=2)
    ax1.set_ylabel('context')

    ax2 = ax1.twinx()
    ax2.plot(range(T), logp_c.squeeze(), label='logp_c', color='blue', linestyle='dotted', linewidth=2)
    ax2.set_ylabel('logp_c')

    fig.legend()

    fig.tight_layout()
    plt.show()



def plot_contexts_rules_states_obs(y_stds, y_dvts, ys, Cs, rules, L_seq, N_seq):
    
    # Visualize tone frequencies
    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax1.plot(y_stds, label='y_std', color='green', linestyle='dotted', linewidth=1, alpha=0.5)
    ax1.plot(y_dvts, label='y_dvt', color='blue', linestyle='dotted', linewidth=1, alpha=0.5)
    ax1.plot(ys, label='y', color='k', marker='o', markersize=2, linestyle='dashed', linewidth=1)
    ax1.set_ylabel('y')

    ax2 = ax1.twinx()
    ax2.plot(Cs, 'o', color='black', label='context', markersize=2)
    ax2.set_ylabel('context')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_yticks(ticks=[0,1], labels=['std', 'dvt'])

    rules_cmap = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:orange'} 
    for i, rule in enumerate(rules):
        plt.axvspan(i*L_seq, i*L_seq + L_seq, facecolor=rules_cmap[rule], alpha=0.25)
        
    for i in range(N_seq):
        plt.axvline(i*L_seq, color='tab:gray', linewidth=0.9)
        ax2.text(x=i*L_seq+0.35*L_seq, y=0.95, s=f'rule {rules[i]}', color=rules_cmap[rules[i]])
        ax2.text(x=i*L_seq+0.35*L_seq, y=0.85, s=f'dvt {dpos[i]}', color=rules_cmap[rules[i]])

    fig.legend(bbox_to_anchor=(1.1, 1)) #, loc='upper left'bbox_to_anchor=(1.05, 1), frameon=False)
    # fig.tight_layout(rect=[0, 0, 0.85, 1])
    # fig.legend()
    plt.tight_layout()
    plt.show()


def plot_rules_dpos(rules, dpos):

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



if __name__=="__main__":

    # Define tone freq LGD parameters
    mu_a = 0.9
    si_a = 0.1
    si_d = 2
    si_q = 2
    si_r = 2


    # w = self._sample_N_(0, self.si_q, len(contexts)-i0-1) # states
    # v = self._sample_N_(0, self.si_r, len(contexts)) # observations
    
    mu_a = 0.85
    si_a = 0.05
    si_d = 5
    sigma_freq_noise = si_q
    sigma_sampling_noise = si_r
    y_values = [1455, 1500, 1600] # Possible freq stationary values



    ######## Choose mode

    # level = 'HC'
    level = 'NHC'


    ######## Non-hierarchical context evolution
    if level == 'NHC':

        # Define number of observations
        T = 50

        # Context initialization and context transition. NOTE: 0: std, 1: dvt
        contexts_set = [0, 1] 
        C0 = np.random.choice(contexts_set)
        C_trans = np.array([[0.9, 0.1], [0.9, 0.1]])

        # Sample pair of frequencies to use 
        y_std, y_dvt = sample_uniform_pair(y_values)

        # Generate observations 
        Cs, ys, y_stds, y_dvts = generate_observations(T_max=T, y_std=y_std, y_dvt=y_dvt, mu_a = mu_a, si_a = si_a, si_d = si_d, sigma_freq_noise=sigma_freq_noise, sigma_sampling_noise=sigma_sampling_noise, C0=C0, C_trans=C_trans, contexts_set=contexts_set)

        plot_contexts_states_obs(Cs, ys, y_stds, y_dvts)

        # Infer with HMM as baseline

        # # Infer with COIN
        # pars = coin.load_pars('validation') # Tune specific parameters later
        # gm = coin.UrnGenerativeModel({'pars': pars, 'name': 'audit_test'})            
        # z_coin, logp_coin, _, lamb = gm.estimate_coin(np.array([ys]), mode='python', nruns=1, max_cores=1)
        # logp_c_coin = compute_logpc(np.array([Cs]), lamb)
        # c_hat = np.argmax(lamb, axis=1)
        
        # plot_contexts_inference(Cs, c_hat, logp_c_coin)   



    ######## What if context is hierachical
    if level == 'HC':

        N_seq = 20 # number of sequences: 4 sequences
        L_seq = 8 # length of each sequence: 8 tones per sequence 
        
        # Rules and rules transitions
        rules_set = [0, 1, 2] 
        R_trans = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]) # A very sticky transition matrix
        rules_dvt_pos = np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]]) # The rule defines the position range of the devt (3-5, 4-6 or 5-7)

        # Timbre
        timbre_set = {0: 'red', 1: ..., 2: ...} # TODO: replace with actual timbre but so far using color
        rules_timbre = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]) # stores proba(i,j) of timbre i for rule j 

        # Initilize rule randomly
        rule = np.random.choice(3)

        # Store rule and sampled dvt_pos for each sequence
        rules = np.zeros(N_seq, dtype=np.int64)
        dpos = np.zeros(N_seq, dtype=np.int64)

        # Store all rules, contexts and observations
        rules_all = np.zeros((N_seq, L_seq))
        Cs = np.zeros((N_seq, L_seq), dtype=np.int64)
        ys = np.zeros((N_seq, L_seq))

        # Store std and dvt dynamics for visualization
        y_stds, y_dvts = np.zeros((N_seq, L_seq)), np.zeros((N_seq, L_seq))

        for n in range(N_seq):
            # Sample pair of frequencies to use 
            y_std, y_dvt = sample_uniform_pair(y_values)

            # We sample the rule and timbre
            rule = sample_markov_state(rule, rules_set, R_trans)
            # timbre = sample_timbre(timbre, timbre_set, rules_timbre)

            # Sample the deviant position
            dvt_pos = sample_dvt_pos(rule, rules_dvt_pos)

            # Store rule and sampled dvt post
            rules[n] = rule
            dpos[n] = dvt_pos

            # Sample params for this sequence
            a_std = _sample_TN_(0, 1, mu_a, si_a, 1).item()
            d_std = _sample_N_(0, si_d, 1)
            a_dvt = _sample_TN_(0, 1, mu_a, si_a, 1).item()
            d_dvt = _sample_N_(0, si_d, 1)

            for t in range(L_seq):
                # Get context for current position (dvt or std)
                ctx = t==dvt_pos # 0 if not at dvt_pos, i.e. std
                
                # Store context and rule
                rules_all[n,t] = rule # Store 
                Cs[n,t] = ctx
                
                # Linear gaussian dynamics of standard and deviant tones
                y_std = y_std * a_std + d_std + sigma_freq_noise * np.random.randn() # TODO: raplce with a sampling at each time step with GM._sample_N_(0, si_r, 1))
                y_dvt = y_dvt * a_dvt + d_dvt + sigma_freq_noise * np.random.randn() # the std and dvt could have different noise variance, TBD later    
                y_stds[n,t] = y_std
                y_dvts[n,t] = y_dvt
                
                # Sample observation
                ys[n,t] = sample_obs(y_std, y_dvt, ctx, sigma_sampling_noise)


        # Flatten sequences
        rules_all = rules_all.flatten()
        Cs = Cs.flatten()
        ys = ys.flatten()
        y_stds = y_stds.flatten()
        y_dvts = y_dvts.flatten()


        # Visualize tone frequencies
        plot_contexts_rules_states_obs(y_stds, y_dvts, ys, Cs, rules, L_seq, N_seq)

        # Visualize hierarchical information: dvt pos and rule
        plot_rules_dpos(rules, dpos)



        pass







