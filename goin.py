import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import coin
import scipy
import glob
import os
import time
import copy
from icecream import ic 
DEV='cuda' # Default device; can be overwritten when instantiating Model and GruRNN


class GruRNN(nn.Module):
	""" 
	Provides for a torch.nn.Module RNN of GRUs with one input (observations) and two sections of 
	outputs: the first encoding sufficient statistics of a Gaussian predictive distribution for the
	observations; the second encoding a discrete predictive distribution for the context.

	Attributes
    ----------
    n_hidden   : int
    	number of hidden units in the RNN
	n_layers   : int
		number of layers in the RNN		
	batch_size : int (optional)
		batch size is used to initialised the RNN on first instantiation; I recommend skipping this
		and explicitly calling init_hidden(batch_size) instead
	train_h0   : bool (optional)
		whether to train or not the initial state of the RNN. If not trained, the hidden state is
		initialised to zero
	dev  : str (optional)
		either 'cpu' or 'cuda'

    Methods
    -------
    init_hidden(batch_size)
    	Initialises the hidden state of the network
	forward(x)
		forward pass of the module; not to be called directly. Use rnn = GruRNN(); out = run(x) 
		instead.
	"""

	def __init__(self, n_hidden=64, n_layers=1, batch_size=None, train_h0=False, dev=DEV):
		
		super(GruRNN, self).__init__()
		
		# Init pars:
		self.dev   = dev
		self.n_hidden = n_hidden
		self.n_layers = n_layers

		# Harcoded pars:
		dropout      = 0.0
		dim_in       = 1
		dim_out_obs  = 2
		dim_out_lamb = 11

		# Networks
		self.gru = nn.GRU(dim_in, n_hidden, n_layers, batch_first=True, dropout=0, device=dev)
		self.out_obs  = nn.Linear(n_hidden, dim_out_obs,  bias=True, device=dev)
		self.out_lamb = nn.Linear(n_hidden, dim_out_lamb, bias=True, device=dev)

		# Setting h0 as an optimisable parameter
		if train_h0:
			h_size = [self.n_layers, 1, self.n_hidden]
			self.h0 = nn.Parameter((torch.randn(*h_size) / (self.n_hidden**.5)).to(self.dev))
		else:
			self.h0 = None

		if batch_size is not None:
			self.init_hidden(batch_size)

	def init_hidden(self, batch_size):
		if self.h0 is None:
			h_size = [self.n_layers, batch_size, self.n_hidden]
			self.hidden_state = torch.zeros(*h_size, requires_grad=False).to(self.dev)
		else:
			self.hidden_state = self.h0.repeat([1, batch_size, 1]) 

	def forward(self, x):
		if self.hidden_state is not None:
			self.init_hidden(batch_size=x.shape[0])
		x, self.hidden_state = self.gru(x, self.hidden_state)
		y = self.out_obs(x)
		l = self.out_lamb(x)
		return y, l


class Model():
	""" 
	Wrapper for a torch.nn.Module (currently tested with GruRNN and transformers only). Provides 
	for interfacing with coin.py, and multiple training and benchmarking methods. All methods 
	that are replicated in the coin GenerativeModel classes have the exact same (or a very similar)
	input specification, except that all np.arrays are torch.tensors.

	Attributes
    ----------
    modelpath  : str (optional)
    	path containing the state dictionary of the model where to load the weights from; if not 
    	specified the weights are randomly initialised by torch.nn 
	sigma_r    : int (optional)
		observation noise of the model; used only during the generation of fictious observations
		for the channel trials
	dev  : str (optional)
		either 'cpu' or 'cuda'
	kwargs  :
		parameters of the nn.Module wrapped by Model() can be additionaly specified as kwargs


    Methods
    -------
	load_weights(modelpath):
		Loads the weights of an existing statedict stored in modelpath to the wrapped nn.Module

	save_weights(modelpath, epoch=None):
		Saves the current statedict of the nn.Module to modelpath. Warning: initial states are not
		saved in the current version (to do).

	call(trg, t0=0):
		Calls the wrapped nn.Module (for internal use only)

	run(x):
		Runs the wrapped nn.Module using x as inputs.

	benchmark(x, c):
		Generates a set of benchmarks (same structure as the equivalent method in 
		coin.GenerativeModel) based on a batch of observations x and contexts c.

	train(parset, savename, oracle=0.1, lr=0.01, train_sched=(25,20), freeze=[], resume=False):
		Trains the wrapped nn.Module to minimise cross entropy on observations (and context IDs) 
		drawn from a coin.GenerativeModel with hyperparametrisation parset

	train_ctx(parset, savename, lr=0.01, train_sched=(5,50), resume=False):
		Trains the wrapped nn.Module to minimise cross entropy on the context IDs drawn from a 
		coin.GenerativeModel with hyperparametrisation parset

	tune(parset, savename=None, oracle=0.1, dweight=0.2, lr=0.01, train_sched=(10,5), freeze=[], resume=False, dataparset=None): 
		Trains the wrapped nn.Module to minimise cross entropy of the predictions on the 
		experimental data corresponding to the dataparset subject. It uses cross-entropy to the 
		predictions on samples drawn from a coin.GenerativeModel with hyperparametrisation parset
		as a regularisation term.

	compute_loss(gm, n_trials, batch_size, ctx=True, dd=None):
		Auxiliary function that computes the three potential terms of the training loss for one 
		batch 

	ctxlossfunc(c, l):
		Auxiliary function that computes the cross-entropy with respect to the context ID for a 
		sequence of ground-truth contexts c and a set of predictions l

	set_optim(lr, train_sched, optimise=None, freeze=None):
		Auxiliary function that sets up the torch optimiser and a learning-rate scheduler before
		training

	set_losslog(savename, resume):
		Auxiliary function that sets up (and loads, if the training is being resumed) the loss 
		history of the optimisation process

	set_tuning_data(dataparset):
		Auxiliary function that loads and prepares the experimental data for training procedures 
		that include fitting the predictions of the model to the actual data

	load_opt_defaults()
		Auxiliary function storing the hardcoded parameters of the optimisation: 1) every how many  
		batches the loss is written down to the loss history; 2) the number of trials in a sample; 
		3) the batch size.

	loss_log(logpath, batch, batch_res, n_batches, tt, lr, loss, losses = (None,None,None)):
		Auxiliary function that stores the loss in a textfile

	training_log(lossHistory, epoch0, batch_res, savename, benchmarks):
		Auxiliary function that prints a summary figure detailing the progress of the training (in
		use when optimising predictions on samples of a coin.GenerativeModel)	

	tuning_log(lossHistory, epoch0, batch_res, savename, benchmarks, dd):
		Auxiliary function that prints a summary figure detailing the progress of the training (in
		use when optimising predictions to match empirical data)	

	run_experiment_coin(experiment, N=20, axs=None):
		Computes the prediction of the wrapped nn.Module under the fields of an experiment from the
		COIN paper

	all_coin_experiments(N=20):
		Computes the predictions of the wrapped nn.Module under the fields for all the experiments 
		of the COIN paper that do not involve cue emissions.

	summary_experiments(savefig=None, N=20, cmap="Greys", axs=None):
		Prints a figure withe the predictions of the wrapped nn.Module for all the experiments of 
		the COIN paper.
	"""

	def __init__(self, modelpath=None, sigma_r=0.1, dev=DEV, **pars):

		self.dev      = dev
		self.sigma_r  = sigma_r
		self.model    = GruRNN(**pars, dev=self.dev)

		if modelpath is not None:
			self.load_weights(modelpath)
		
	def load_weights(self, modelpath):
	    """Loads the weights of an existing statedict stored in modelpath to the wrapped nn.Module

	    Parameters
	    ----------
	    modelpath : str (optional)
	    	Path to an existing saved statedict. If not a full path, it assumes the file is in
	    	./models. 
	    """

		if '.pt' not in modelpath:
			modelpath += '.pt'
		if modelpath[0] != '/':
			if './models' not in modelpath:
				modelpath = './models/' + modelpath
		self.model.load_state_dict(torch.load(modelpath))
		self.model.gru.flatten_parameters() 

	def save_weights(self, modelpath, epoch=None):
		""" Saves the current statedict of the nn.Module to modelpath. Warning: initial states ar
			not saved in the current version (to do).

	    Parameters
	    ----------
	    modelpath : str
	    	Path where to save the statedict. If not a full path, it assumes the file is in
	    	./models.
	    epoch     : int (optional)
			If provided, it appends 'e{epoch}' to modelpath; useful to save the training progress.
	    """

		if epoch is not None:
			modelpath = f'{modelpath.split(".")[0]}_e{epoch:02d}'
		if '.pt' not in modelpath:
			modelpath += '.pt'
		if modelpath[0] != '/':
			if './models' not in modelpath:
				modelpath = './models/' + modelpath

		torch.save(self.model.state_dict(), modelpath)

	# Running
	def call(self, trg, t0=0):
		""" Internal function that calls the wrapped nn.Module (to do: rename to _call_)

	    Parameters
	    ----------
	    trg : torch.tensor
	    	Input tensor with three dimensions: dim0 runs across the batch, dim1 runs across 
	    	timepoints (i.e., trials in the COIN jargon), dim2 runs across input dimensions (set to
	    	one in this library).
		t0  : time-point to which trg[:, 0, 0] corresponds; useful to evaluate the model over 
			arbitrary time-segments (necessary to handle channel trials). T0 needs to be specified
			because predictions on contexts are constrained so that max_n_ctx = current_t

	    Returns
	    ----------
		mu  : torch.tensor
			Mean of the predictions; not shifted (i.e., mu[:, t] are the predictions for t+1). 
			First dimension runs across the batch, second across time-points.
		si  : torch.tensor
			Standard deviation of the predictions; same shape and shifting as mu
		lambhat  : torch.tensor
			Distributions for the predictions on the contexts; first two dimensions as mu, last 
			dimensions runs across contexts. 
	    """

		obs, lamb = self.model(trg)
		mu      = obs[..., [0]]
		sigma   = torch.log(1 + torch.exp(obs[..., [1]]))

		# Next three lines constraint context prediction during the first max_ctx trials so that
		# there cannot be more than n context at time-point t=n
		lambmask = torch.ones(lamb.shape).to(self.dev).float()
		for t in range(t0+1, min(lamb.shape[1:3])):
			lambmask[:, t-(t0+1), t+1:] = 1E-30

		lambhat = torch.log((1 + torch.exp(lamb)) * lambmask)
		lambhat = lambhat - torch.logsumexp(lambhat, dim=2, keepdim=True)
		return (mu, sigma, lambhat)

	def run(self, x):
		""" Runs the wrapped nn.Module using x as inputs.

	    Parameters
	    ----------
	    x : torch.tensor
	    	Input tensor enconding sequences of observations; three dimensions: dim0 runs across the
	    	batch, dim1 runs across timepoints (i.e., trials in the COIN jargon), dim2 set to one.

	    Returns
	    ----------
		u  : torch.tensor
			Mean of the predictions; shifted (i.e., u[:, t] are the predictions for t). 
			First dimension runs across batches, second across time-points.
		s  : torch.tensor
			Standard deviation of the predictions; same shape and shifting as u
		l  : torch.tensor
			Distributions for the predictions on the contexts; first two dimensions as u, last 
			dimensions runs across contexts. 
	    """
		
		n_batches, seq_len, dim = x.shape
		n_o = 1 if self.model.out_obs.out_features == 2 else int(self.model.out_lamb.out_features/3)
		n_c = self.model.out_lamb.out_features

		if isinstance(self.model, GruRNN):
			self.model.init_hidden(n_batches)

		u = (torch.zeros((n_batches, seq_len, n_o), requires_grad=False) * torch.nan).to(self.dev)
		s = (torch.zeros((n_batches, seq_len, n_o), requires_grad=False) * torch.nan).to(self.dev)
		l = (torch.zeros((n_batches, seq_len, n_c), requires_grad=False) * torch.nan).to(self.dev)
		l[:, 0, 1:] = -30
		l[:, 0, 0]  = 0

		if torch.any(torch.isnan(x)) and isinstance(self.model, GruRNN):
			for b in range(n_batches):
				chantrials = torch.where(torch.any(torch.isnan(x[b, :, :]), 1))[0]
				t0 = 0
				for t1 in chantrials:
					u[b, (t0+1):(t1+1), :], s[b, (t0+1):(t1+1), :], l[b, (t0+1):(t1+1), :] = self.call(x[[b], t0:t1, :])
					x[b, t1, :] = u[b, t1, :] + self.sigma_r * torch.randn(dim).to(self.dev)
					t0 = copy.deepcopy(t1)
		else:
			u[:, 1:, :], s[:, 1:, :], l[:, 1:, :] = self.call(x[:, :-1, :])

		return u, s, l

	def benchmark(self, x, c):
		""" Generates a set of benchmarks (same structure as the equivalent method in 
			coin.GenerativeModel) based on a batch of observations x and contexts c.

	    Parameters
	    ----------
	    x : torch.tensor
	    	Input tensor enconding sequences of observations; three dimensions: dim0 runs across the
	    	batch, dim1 runs across timepoints (i.e., trials in the COIN jargon), dim2 set to one.
	    c : torch.tensor 
	    	Input tensor encoding the sequences of contexts associated to the observations in x; 
	    	same shape as x.

	    Returns
	    ----------
		perf  : dict
			Summary statistics (mse, KS-statistic, and cross-entropy of the predictions for the 
			observations; accuracy and cross-entropy for the contexts) measuring the performance of
			the model for the set of observations specified in the input.
		u     : torch.tensor
			Mean of the predictions; shifted (i.e., u[:, t] are the predictions for t). First 
			dimension runs across batches, second across time-points.
		s     : torch.tensor
			 Standard deviation of the predictions; same shape and shifting as u
		c_hat : torch.tensor
			Predicted context; same shape and shifting as u
		ct_pt : torch.tensor
			Probability of the predicted context; same shape and shiftings as u
	    """

		nb, nt = x.shape[0], x.shape[1]
		u, s = np.zeros(x.shape), np.zeros(x.shape) 
		l = np.zeros((x.shape[0], x.shape[1], self.model.out_lamb.out_features))

		bunches = [(64*n, min(64*(n+1), x.shape[0])) for  n in range(int(np.ceil(x.shape[0]/64)))]
		for b in bunches:
			z = torch.tensor(x[b[0]:b[1]], dtype=torch.float, requires_grad=False).to(self.dev)
			with torch.no_grad():
				bu, bs, bl = self.run(z)
			u[b[0]:b[1], :, :] = bu.cpu().detach().numpy()
			s[b[0]:b[1], :, :] = bs.cpu().detach().numpy()
			l[b[0]:b[1], :, :] = bl.cpu().detach().numpy()

		# Observations
		mse = (u - x)**2
		llk = -0.5 * np.log(2*np.pi) -np.log(s) - 0.5 * ((u - x) / s) ** 2
		KS  = measure_KS_stat(x, u, s)
	
		# Context 
		c_hat = np.argmax(l, axis=2)
		ct_ac = (c_hat == c[..., 0]).mean(1)


		lamb = np.exp(l)
		ct_pr = np.zeros(c.shape[:2])
		ct_ce = np.zeros(x.shape[0])

		for b in range(c.shape[0]):
			ct_pr[b] = np.array([lamb[b, t, c_hat[b, t]] for t in range(c.shape[1])])

		for b in range(c.shape[0]):
			for ctx in range(l.shape[2]):
				ctx_ix = np.where(c[b, :, 0] == ctx)[0]
				ct_ce[b] += np.nansum(l[b, ctx_ix, ctx]) / c.shape[1]

		# Storing
		perf = {'mse': {}, 'kol': {}, 'ce': {}, 'ct_ac': {}, 'ct_p': {}, 'ct_ce': {}}

		perf['mse']['avg'] = np.nanmean(mse)
		perf['mse']['sem'] = np.nanstd(mse) / np.sqrt(nb)
		perf['ce']['avg']  = np.nanmean(llk)
		perf['ce']['sem']  = np.nanstd(llk) / np.sqrt(nb)
		perf['kol']['avg'] = np.nanmean(KS)
		perf['kol']['sem'] = np.nanstd(KS) / np.sqrt(nb)

		perf['ct_ac']['avg'] = np.nanmean(ct_ac)
		perf['ct_ac']['sem'] = np.nanstd(ct_ac) / np.sqrt(nb)
		perf['ct_ce']['avg']  = np.nanmean(ct_ce)
		perf['ct_ce']['sem']  = np.nanstd(ct_ce) / np.sqrt(nb)

		return perf, u, s, c_hat, ct_pr

	# Training
	def train(self, parset, savename, oracle=0.1, lr=0.01, train_sched=(25,20), freeze=[], resume=False):
		""" Trains the wrapped nn.Module to minimise cross entropy on observations (and context IDs) 
			drawn from a coin.GenerativeModel with hyperparametrisation parset.
	    
	    Parameters
	    ----------
		parset
	    	String identifying the hyperparametrisation of the coin.GenerativeModel with which to 
	    	train the model. See coin.load_pars() for more details.
	    savename : str 
	    	Path where to save the statedict. If not a full path, it assumes the file is in
	    	./models.
		oracle   : float (optional)
			Normalised weight of the component of the loss associated to the context cross-entropy.
			I.e., the training loss = (1-oracle) * loss_observations + oracle * loss_context. 
			0 <= oracle <= 1.
		lr  : float (optional)
			Baseline learning rate.
		train_sched : tuple
			Tuple specifying the (maximum number of epochs, number of batches in an epoch). The 
			number of training epochs will be shorter if the MSE of the model reaches the COIN 
			benchmark at an earlier time.
		freeze  : list (optional)
			List of string indicating the modules of the warapped nn.modules that should not be 
			modified during training. Each string should be the name of a submodule of the model, 
			not the specific set of weights; e.g., 'gru', 'out_obs', or 'out_lamb' when using GruRNN
		resume  : bool (optional)
			Whether to resume training or start from scratch. If set to True, it will attempt to 
			locate an existing loss_log.
	    """

		# Hardcoded parameters
		n_trials, batch_size, batch_res = self.load_opt_defaults()
		tolerance = 0.001 # Number of SEMs above COIN performance for early training halting

		# Setting up optimisation
		self.model.train()
		n_epochs, n_batches, opt, lr_scheduler = self.set_optim(lr, train_sched, freeze=freeze)
		lossfunc = nn.GaussianNLLLoss()

		# Preparing/loading loss log
		lossHistory, epoch0, logpath = self.set_losslog(savename, resume)

		# Setting up GM and benchmarks
		gm = coin.CRFGenerativeModel(parset)
		benchmarks = gm.benchmark(n_trials, suffix='training')
		mse_target = benchmarks['perf']['coin']['mse']['avg']
		mse_target += tolerance * benchmarks['perf']['coin']['mse']['sem']

		# Training
		for e in range(epoch0, epoch0 + n_epochs):

			epocht0 = time.time()

			print(f'### TRAINING EPOCH {e} ({gm.parset})  | {time.ctime()} ###')
			tt = time.time()
			lossHistory.append([])

			for batch in range(n_batches):

				# Initialise model
				self.model.init_hidden(batch_size)
				opt.zero_grad()

				# Compute loss
				loss_obs, loss_ctx, _ = self.compute_loss(gm, n_trials, batch_size, ctx=(oracle!=0))
				loss = (1.-oracle) * loss_obs + oracle * loss_ctx

				# Step
				loss.backward()
				opt.step()
				lr_scheduler.step()

				# Logging and reporting
				if batch % batch_res == batch_res-1:
					lr = lr_scheduler.get_last_lr()[0]
					losses  = (loss_obs, loss_ctx, None) if oracle!=0 else (loss_obs, None, None)
					loss_np = self.loss_log(logpath,batch,batch_res,n_batches,tt,lr,loss,losses)
					tt = time.time()
					lossHistory[-1].append(loss_np)

			if savename is not None:
				self.save_weights(savename, epoch=e)
			
			print(f' ----- Epoch {e} done! Epoch time = {(time.time()-epocht0)/60:.1f} minutes')
			mse = self.training_log(lossHistory, epoch0, batch_res, savename, benchmarks)[0]

			if mse < mse_target:
				mse_target_s  = f'{benchmarks["perf"]["coin"]["mse"]["avg"]:.3f}'
				mse_target_s += f'+{tolerance:.3f} x {benchmarks["perf"]["coin"]["mse"]["sem"]:.3f}'
				print(f'MSE target reached ({mse:.3f} < {mse_target_s})!')
				break

		print('###### END OF TRAINING ######\n')

	def train_ctx(self, parset, savename, lr=0.01, train_sched=(5,50), resume=False):
		""" Trains the lamd_out section of the wrapped nn.Module to minimise cross entropy on the 
			context IDs drawn from a coin.GenerativeModel with hyperparametrisation parset

	    Parameters
	    ----------
		parset
	    	String identifying the hyperparametrisation of the coin.GenerativeModel with which to 
	    	train the model. See coin.load_pars() for more details.
	    savename : str 
	    	Path where to save the statedict. If not a full path, it assumes the file is in
	    	./models.
		lr  : float (optional)
			Baseline learning rate.
		train_sched : tuple
			Tuple specifying the (maximum number of epochs, number of batches in an epoch). The 
			number of training epochs will be shorter if the MSE of the model reaches the COIN 
			benchmark at an earlier time.
		resume  : bool (optional)
			Whether to resume training or start from scratch. If set to True, it will attempt to 
			locate an existing loss_log.
	    """

		# Hardcoded paramters
		n_trials, batch_size, batch_res = self.load_opt_defaults()
		tolerance = 0.01 # Number of SEMs above COIN ctx accuracy for early training halting

		# Setting up optimisation
		self.model.train()
		n_epochs,n_batches,opt,lr_scheduler = self.set_optim(lr, train_sched, optimise='out_lamb')

		# Preparing/loading loss log
		lossHistory, epoch0, logpath = self.set_losslog(savename, resume)

		# Setting up GM and benchmarks
		gm = coin.CRFGenerativeModel(parset)
		benchmarks = gm.benchmark(n_trials, suffix='training')
		acc_target = benchmarks['perf']['coin']['ct_ac']['avg']
		acc_target += tolerance * benchmarks['perf']['coin']['ct_ac']['sem']

		# Training
		for e in range(epoch0, epoch0 + n_epochs):

			epocht0 = time.time()

			print(f'### CTX-TRAINING EPOCH {e} ({gm.parset}) | {time.ctime()} ###')
			tt = time.time()
			lossHistory.append([])

			for batch in range(n_batches):

				# s0 = dict([(key, self.model.state_dict()[key].clone().detach()) for key in self.model.state_dict()])
				# Initialise model
				self.model.init_hidden(batch_size)
				opt.zero_grad()

				# Generate data
				y, q, c = gm.generate_batch(n_trials, batch_size)
				x = torch.tensor(y, dtype=torch.float, requires_grad=False).to(self.dev)

				# Compute loss
				_, loss, _ = self.compute_loss(gm, n_trials, batch_size, ctx=True)

				# Step
				loss.backward()
				opt.step()
				lr_scheduler.step()

				# print(dict([(key, torch.all(s0[key]==self.model.state_dict()[key]).item()) for key in s0]))

				# Logging and reporting
				if batch % batch_res == batch_res-1:
					lr = lr_scheduler.get_last_lr()[0]
					losses  = (None, loss, None)
					loss_np = self.loss_log(logpath,batch,batch_res,n_batches,tt,lr,loss,losses)
					tt = time.time()
					lossHistory[-1].append(loss_np)
			
			if savename is not None:
				self.save_weights(savename, epoch=e)

			print(f' ----- Epoch {e} done! Epoch time = {(time.time()-epocht0)/60:.1f} minutes')
			acc = self.training_log(lossHistory, epoch0, batch_res, savename, benchmarks)[1]

			if acc > acc_target:
				acc_target_s  = f'{benchmarks["perf"]["coin"]["ct_ac"]["avg"]:.3f}'
				acc_target_s += f'+{tolerance:.3f} x {benchmarks["perf"]["coin"]["ct_ac"]["sem"]:.3f}'
				print(f'Accuracy target reached ({acc:.3f} > {acc_target_s})!')
				break

		print(f'###### END OF CTX-TRAINING TRAINING ######\n')

	def tune(self, parset, savename=None, oracle=0.1, dweight=0.2, lr=0.01, train_sched=(10,5), freeze=[], resume=False, dataparset=None): 
		""" Trains the wrapped nn.Module to minimise cross entropy of the predictions on the 
		experimental data corresponding to the dataparset subject. It uses cross-entropy to the 
		predictions on samples drawn from a coin.GenerativeModel with hyperparametrisation parset
		as a regularisation term.

	    Parameters
	    ----------
		parset
	    	String identifying the hyperparametrisation of the coin.GenerativeModel with which to 
	    	train the model. See coin.load_pars() for more details.
	    savename : str 
	    	Path where to save the statedict. If not a full path, it assumes the file is in
	    	./models.
		oracle   : float (optional)
			Normalised weight of the component of the regularisation term of the loss associated to
			the context cross-entropy. I.e.:
			regularisation_loss = (1-oracle) * loss_observations + oracle * loss_context. 
			0 <= oracle <= 1.
		dweight  : float (optional)
			Normalised weight of the component of the loss associated to the data fit.
			I.e., the training loss = dweight * loss_data + (1-dweithg) * regularisation_loss
			0 <= dweight <= 1.
		lr  : float (optional)
			Baseline learning rate.
		train_sched : tuple
			Tuple specifying the (maximum number of epochs, number of batches in an epoch). The 
			number of training epochs will be shorter if the MSE of the model reaches the COIN 
			benchmark at an earlier time.
		freeze  : list (optional)
			List of string indicating the modules of the warapped nn.modules that should not be 
			modified during training. Each string should be the name of a submodule of the model, 
			not the specific set of weights; e.g., 'gru', 'out_obs', or 'out_lamb' when using GruRNN
		resume  : bool (optional)
			Whether to resume training or start from scratch. If set to True, it will attempt to 
			locate an existing loss_log.
		dataparset  : str
		    String identifying the dataset to use to train the model. See coin.load_recovery_data()
		    for more info. If dataparset is not specified, it is set to coincide with parset.
	    """

		if dataparset is None:
			dataparset = parset

		# Hardcoded paramters
		n_trials, batch_size, batch_res = self.load_opt_defaults()
		tolerance = 0.5  # minimum MSE of the model for early tuning halting as number of SEMs

		# Setting up optimisation
		self.model.train()
		n_epochs,n_batches,opt,lr_scheduler = self.set_optim(lr, train_sched, freeze=freeze)

		# Preparing/loading loss log
		lossHistory, epoch0, logpath = self.set_losslog(savename, resume)

		# Setting up GM and benchmarks
		gm = coin.CRFGenerativeModel(parset)
		benchmarks = gm.benchmark(n_trials, suffix='training')
		dd = self.set_tuning_data(dataparset)

		# Tuning
		for e in range(epoch0, epoch0 + n_epochs):

			epocht0 = time.time()

			print(f'### TUNING EPOCH {e} ({gm.parset})  | {time.ctime()} ###')
			tt = time.time()
			lossHistory.append([])

			for batch in range(n_batches):

				# Initialise model
				self.model.init_hidden(batch_size)
				opt.zero_grad()

				# Compute loss
				ctx = (oracle!=0)
				loss_obs,loss_ctx,loss_dd = self.compute_loss(gm, n_trials, batch_size, ctx, dd)
				loss = (1-dweight) * ((1-oracle)*loss_obs + oracle*loss_ctx) + dweight * loss_dd

				# Step
				loss.backward()
				opt.step()
				lr_scheduler.step()

				# Logging and reporting
				if batch % batch_res == batch_res-1:
					lr = lr_scheduler.get_last_lr()[0]
					losses  = (loss_obs,loss_ctx,loss_dd) if oracle !=0 else (loss_obs,None,loss_dd)
					loss_np = self.loss_log(logpath,batch,batch_res,n_batches,tt,lr,loss,losses)
					tt = time.time()
					lossHistory[-1].append(loss_np)
			
			if savename is not None:
				self.save_weights(savename, epoch=e)

			print(f' ----- Epoch {e} done! Epoch time = {(time.time()-epocht0)/60:.1f} minutes')
			mse_r = self.tuning_log(lossHistory, epoch0, batch_res, savename, benchmarks, dd)

			if mse_r < tolerance:
				mse_target_s  = f'{benchmarks["perf"]["coin"]["mse"]["avg"]:.3f}'
				mse_target_s += f'+{tolerance:.3f} x {benchmarks["perf"]["coin"]["mse"]["avg"]:.3f}'
				print(f'MSE target reached (mse/sem = {mse_r:.2f} < {tolerance:.2f})!')
				break

		print('###### END OF TUNING ######\n')

	def compute_loss(self, gm, n_trials, batch_size, ctx=True, dd=None):
		""" Auxiliary function that computes the three potential terms of the training loss for one 
		batch

	    Parameters
	    ----------
	    gm         : coin.GenerativeModel
	    	A GenerativeModel object to draw data from. CRFGenerativeModel runs faster for low 
	    	n_trials.
	    n_trials   : int
	    	number of time points (i.e., trials in the COIN jargon) in each sequence
	    batch_size : int
	    	number of batches
		ctx  :  bool
			whether to compute or not context cross-entropy
		dd   :  dict
			set encoding the experimental data for the fit, as computed by set_tuning_data()

	    Returns
	    ----------
	    loss_obs  : scalar torch.tensor
	    	training loss corresponding to cross-entropy of the predictions for the observations
	    loss_ctx  : scalar torch.tensor
	    	training loss corresponding to cross-entropy of the context predictions
	    loss_dd   : scalar torch.tensor
	    	training loss corresponding to cross-entropy of the data fir
	    """

		lossfunc = nn.GaussianNLLLoss()

		y, q, c = gm.generate_batch(n_trials, batch_size)
		x = torch.tensor(y, dtype=torch.float, requires_grad=False).to(self.dev)
		u, s, l = self.run(x)

		loss_obs = lossfunc(x[:,1:,0], u[:,1:,0], s[:,1:,0]**2)
		loss_ctx = self.ctxlossfunc(c, l) if ctx else torch.tensor(0)

		if dd is None:
			loss_dd = torch.tensor(0)
		else:
			noise = gm.si_r * torch.randn(dd['f'].shape).to(self.dev).float()
			u, s, c = self.run(dd['f'] + noise)
			loss_dd = lossfunc(dd['y'], u[0, dd['t'], 0], s[0, dd['t'], 0]**2)

		return loss_obs, loss_ctx, loss_dd

	def ctxlossfunc(self, c, l):
		""" Auxiliary function that computes the cross-entropy with respect to the context ID for a 
		sequence of ground-truth contexts c and a set of predictions l
		
	    Parameters
	    ----------
	    c : torch.tensor 
	    	Input tensor encoding the ground-truth sequences of contexts; dim0 runs across the
	    	batch, dim1 runs across timepoints (i.e., trials in the COIN jargon), dim2 set to one.
	    l : torch.tensor 
	    	Input tensor encoding the distribution of the predictions for the contexts; dims 0-1 
	    	same as c; dim2 runs across the possible contexts

	    Returns
	    scalar torch.tensor
	    	training loss corresponding to cross-entropy of the context predictions
	    ----------
	    """

		cflat = c[...,0].reshape(-1)
		lflat = l.reshape(-1, l.shape[2])
		loss = -torch.stack([torch.nansum(lflat[cflat==ctx, ctx]) for ctx in range(l.shape[2])]).sum()
		return loss / len(cflat)

	def set_optim(self, lr, train_sched, optimise=None, freeze=None):
		""" Auxiliary function that sets up the torch optimiser and a learning-rate scheduler before
		training
		
	    Parameters
	    ----------
		lr  : float (optional)
			Baseline learning rate.
		train_sched : tuple
			Tuple specifying the (maximum number of epochs, number of batches in an epoch). The 
			number of training epochs will be shorter if the MSE of the model reaches the COIN 
			benchmark at an earlier time.
		optimise: list (optional)
			List of string indicating the modules of the warapped nn.modules that SHOULD be 
			modified during training. Each string should be the name of a submodule of the model, 
			not the specific set of weights; e.g., 'gru', 'out_obs', or 'out_lamb' when using GruRNN 
		freeze  : list (optional)
			List of string indicating the modules of the warapped nn.modules that should NOT be 
			modified during training. Each string should be the name of a submodule of the model, 
			not the specific set of weights; e.g., 'gru', 'out_obs', or 'out_lamb' when using GruRNN
			Ignored if optimise is specified. 
	    Returns
	    ----------
	    n_epochs     : int
	    	maximum number of epochs
	    n_batches    : int
			number of batches in each epoch
	    opt          : torch.optim.Adam
	    	optimiser for the training procedure
	    lr_scheduler : torch.optim.lr_scheduler
	    	learning rate scheduler for the training procedure
	    """

		# Hardcoded parameters
		weight_decay = 1e-5 

		optimise_pars = []
		if optimise is not None:
			if type(optimise) is not list:
				optimise = [optimise]
			if freeze is None:
				freeze = []
			for name, para in self.model.named_parameters():
				if name.split('.')[0] not in optimise and name.split('.')[0] not in freeze:
					freeze.append(name.split('.')[0])
		
		for name, para in self.model.named_parameters():
			if name.split('.')[0] in freeze:
				print(f'Freezing {name}')
				para.requires_grad = False
			else:
				para.requires_grad = True
				optimise_pars.append(para)

		n_epochs, n_batches = train_sched
		opt = optim.Adam(optimise_pars, lr=lr, weight_decay=weight_decay) 
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=np.prod(train_sched))

		return n_epochs, n_batches, opt, lr_scheduler

	def set_losslog(self, savename, resume):
		""" Auxiliary function that sets up (and loads, if the training is being resumed) the loss 
		history of the optimisation process

	    Parameters
	    ----------
	    savename : str 
	    	Path where the statedict of the model will be saved; used to derive a saving path for 
	    	the loss log. 
		resume  : bool (optional)
			Whether to resume training or start from scratch. If set to True, it will attempt to 
			locate an existing loss_log.
	    Returns
	    ----------
	    """

		logpath = f'./logs/{savename.split(".")[0]}-losslog'
		if resume: # Resume not tested after extending losslog to four sub-losses
			loadedLoss = []
			with open(logpath, 'r') as f:
				for line in [lin for lin in f.read().split("\n") if lin !='']:
					loadedLoss.append(float(line.replace('\t', ',').split(',')[0]))
			loadedLoss = np.array(loadedLoss)
			epoch0 = int(np.round(len(loadedLoss) / (n_batches/batch_res)))
			n_points = int(len(loadedLoss) / epoch0)
			lossHistory = [list(loadedLoss[(n0*n_points):((n0+1)*n_points)]) for n0 in range(epoch0)]
		else:
			if os.path.exists(logpath):
				os.remove(logpath)
			lossHistory, epoch0 = [], 0

		return lossHistory, epoch0, logpath

	def set_tuning_data(self, dataparset):
		""" Auxiliary function that loads and prepares the experimental data for training procedures 
			that include fitting the predictions of the model to the actual data

	    Parameters
	    ----------
		dataparset  : str
		    String identifying the dataset to use to train the model. See coin.load_recovery_data()
		    for more info.
	    
	    Returns
	    dict
	    	Dictionary with fields: 
			dd['f'] : sequence of ground-truth field of the experiments
			dd['t'] : timepoints corresponding to each of the datapoints 
			dd['y'] : datapoints
			dd['x'] : noisy observations of the ground-truth fields to serve as model inputs
			dd['u'] : predictions of the COIN model for dd['x']
	    ----------
	    """

		gm   = coin.CRFGenerativeModel(dataparset)
		data = coin.load_recovery_data(dataparset)

		t = data[1].astype(np.int32)
		x = (data[2].astype(float) + gm.si_r * np.random.randn(20, data[2].shape[0]))[..., None]
		u_coin = gm.estimate_coin(x)[0][:, t]

		dd = {'f' : torch.tensor(data[2][None, :, None]).to(self.dev).float(),
			  't' : t,
			  'y' : torch.tensor(data[0][0]).to(self.dev).float(),
			  'x' : x,
			  'u' : u_coin}

		return dd

	def load_opt_defaults(self):
		""" Auxiliary function storing the hardcoded parameters of the optimisation
	    
	    Returns
	    ----------
	    n_trials   : int
	    	number of time points (i.e., trials in the COIN jargon)
	    batch_size : int
	    	number of batches
	    batch_res  : int
	    	every how many batches the loss is written down to the loss history
	    """

		n_trials     = 5000
		batch_res    = 1    # Store and report loss every batch_res batches
		batch_size   = 64

		return n_trials, batch_size, batch_res
	
	def loss_log(self, logpath, batch, batch_res, n_batches, tt, lr, loss, losses = (None,None,None)):
		""" Auxiliary function that stores the loss in a textfile

	    Parameters
	    ----------
	    logpath   : str (optional)
	    	Path of the textfile where to log the loss.
		batch     : int
			Current batch number
	    batch_res : int
	    	Every how many batches the loss is written down to the loss history
		n_batches : int
	    	Number of batches
		tt        : 
			Time at which the training for the current batch stated (as measured by time.time()) 
		lr        : float
			current learning rate
		loss      : scalar torch.tensor
			total loss
		losses    : tuple of thre scalar torch.tensors 
		 	encoding (observation loss, context loss, datafit loss); use None for the terms of the
		 	loss that are not in use.

	    Returns
	    ----------
	    loss_np : scalar np.array
	    	A detached version of the total loss for reporting 
	    """

		loss_obs, loss_ctx, loss_dd = losses

		loss_np = loss.detach().item()
		sprint  = f'Batch {batch+1:>2}/{n_batches}; Time = {time.time()-tt:.1f}s; '
		sprint += f'Loss = {loss_np:.3f} ('
		slog = f'{loss_np/batch_res:.2f}'
		
		for l, n in zip(losses, ('obs', 'ctx', 'data')):
			if l is not None:
				l_np = l.detach().item()
				slog += f',{l_np/batch_res:.2f}' 
				sprint += f'{n} = {l_np:.3f}, '
			else:
				slog += f',0.00'
				sprint += f'{n} = ----, '
		sprint = f'{sprint[:-2]}); '

		sprint += f'LR = {lr:.2g})'

		with open(logpath, 'a') as f:
			f.write(f'{slog}\n')

		print(sprint)

		return loss_np

	def training_log(self, lossHistory, epoch0, batch_res, savename, benchmarks):
		""" Auxiliary function that prints a summary figure detailing the progress of the training (in
			use when optimising predictions on samples of a coin.GenerativeModel)
	    
	    Parameters
	    ----------
		lossHistory  : list
			List of lists where each list encodes the history of the loss function in each epoch
		epoch0       : int
			The number of the epoch in which the training has started (useful for resumed training)
	    batch_res    : int
	    	Every how many batches the loss is written down to the loss history
		savename     : str
			Path where the statedict of the model will be saved; used to derive a saving path for 
	    	the summary figure
		benchmarks   : dict
			Dictionary with the benchmarks for the generative model, as provided by 
			coin.GenerativeModel.benchmark()

	    Returns
	    ----------
	    mse_avg  : float
	    	Mse of the model to the batch of samples stored in benchmarks
	    mse_std  : float
	    	Standard deviation of the model predictions
	    """

		# Estimating model predictions on the benchmarking samples
		self.model.eval()
		X, C = benchmarks['X'], benchmarks['C']
		perf, u, s, c_hat, c_pr = self.benchmark(X, C)
		self.model.train()

		# Figure config
		fig  = plt.figure()
		gs   = fig.add_gridspec(4,6)
		ax_ex = [fig.add_subplot(gs[n, :]) for n in range(3)]
		ax_mse, ax_kol  = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
		ax_cacc, ax_cce = fig.add_subplot(gs[3, 2]), fig.add_subplot(gs[3, 3])
		ax_loss = fig.add_subplot(gs[3, 4:]) 

		# Plotting a few examples
		for n in range(len(ax_ex)):
			plot_predictions(X[n], u[n], s[n], C[n], c_hat[n], c_pr[n],  ax_ex[n])
			

		# Plotting Performance
		perf = {**{'gru': perf}, **benchmarks['perf']}
		plotvec = dict()
		for key in ['mse', 'kol', 'ce']:
			plotvec[key] = dict()
			for val in ['avg', 'sem']:
				plotvec[key][val] =  [perf[mod][key][val] for mod in ['LI', 'coin', 'gru']]

		for key in ['ct_ac', 'ct_ce']:
			plotvec[key] = dict()
			for val in ['avg', 'sem']:
				plotvec[key][val] =  [perf[mod][key][val] for mod in ['coin', 'gru']]

		colours = ['tab:purple', 'tab:green', 'tab:gray'] 
		ax_mse.bar([0, 1, 2], plotvec['mse']['avg'], yerr=plotvec['mse']['sem'], color=colours)
		ax_mse.set_xticks([0, 1, 2])
		ax_mse.set_xticklabels(['Leak-Int', 'COIN', 'GRU'])
		ax_mse.set_ylabel('mse')
		ax_kol.bar([0, 1, 2], plotvec['kol']['avg'], yerr=plotvec['kol']['sem'], color=colours)
		ax_kol.set_xticks([0, 1, 2])
		ax_kol.set_xticklabels(['Leak-Int', 'COIN', 'GRU'])
		ax_kol.set_ylabel('KS-stat')
		
		colours = ['tab:green', 'tab:gray'] 
		ax_cacc.bar([0, 1], plotvec['ct_ac']['avg'], yerr=plotvec['ct_ac']['sem'], color=colours)
		ax_cacc.set_xticks([0, 1])
		ax_cacc.set_xticklabels(['COIN', 'GRU'])
		ax_cacc.set_ylabel('ctx accuracy')
		ax_cce.bar( [0, 1], plotvec['ct_ce']['avg'],  yerr=plotvec['ct_ce']['sem'],  color=colours)
		ax_cce.set_xticks([0, 1])
		ax_cce.set_xticklabels(['COIN', 'GRU'])
		ax_cce.set_ylabel('ctx CE')

		# Plotting loss evolution
		t0 = 0
		colours = ['k', 'tab:gray']
		for e in range(len(lossHistory)):
			trials = batch_res * np.arange(t0, t0 + len(lossHistory[e]))
			ax_loss.plot(trials, np.array(lossHistory[e]), color=colours[(epoch0+e)%2])
			t0 += len(lossHistory[e])
		ymin = 0.2 * np.floor(5 * np.array(lossHistory).min()) - 0.2
		ymax = 0.2 * min(25, 5 * np.ceil(np.array(lossHistory).max())) + 0.1
		ax_loss.set_ylim([ymin, ymax])
		ax_loss.set_xlabel('batch number')
		ax_loss.set_ylabel('loss')

		# Marking last 5 minima
		lharray = np.array(lossHistory)
		ix = np.unravel_index(lharray.reshape(-1).argsort()[:5], lharray.shape)
		for e, b, n in zip(*ix, range(len(ix[0]))):
			trial = batch_res * (sum([len(lossHistory[e0]) for e0 in range(e)]) + b)
			c0 = np.array(matplotlib.colors.to_rgb('tab:red'))
			c1 = np.array(matplotlib.colors.to_rgb('tab:blue'))
			c1 = np.array([255, 239, 69]) / 255
			c0 = np.array([255, 0, 0]) / 255
			cTrial = matplotlib.colors.to_hex((1-(n/(len(ix[0])-1)))*c0 + (n/(len(ix[0])-1))*c1)
			ax_loss.plot(trial, lossHistory[e][b], 'v', color=cTrial)

		# Saving
		fig.set_size_inches(23, 13)
		fig.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.3,hspace=0.3)
		plt.savefig(f'./logs/{savename.split(".")[0]}-e{len(lossHistory):03}.png')
		plt.cla()
		plt.close(fig)

		return(perf['gru']['mse']['avg'], perf['gru']['ct_ac']['avg'])
		
	def tuning_log(self, lossHistory, epoch0, batch_res, savename, benchmarks, dd):
		""" Auxiliary function that prints a summary figure detailing the progress of the training (in
			use when optimising predictions to match empirical data)	

	    Parameters
	    ----------
		lossHistory  : list
			List of lists where each list encodes the history of the loss function in each epoch
		epoch0       : int
			The number of the epoch in which the training has started (useful for resumed training)
	    batch_res    : int
	    	Every how many batches the loss is written down to the loss history
		savename     : str
			Path where the statedict of the model will be saved; used to derive a saving path for 
	    	the summary figure
		benchmarks   : dict
			Dictionary with the benchmarks for the generative model, as provided by 
			coin.GenerativeModel.benchmark()
		dd   :  dict
			set encoding the experimental data for the fit, as computed by set_tuning_data()

	    Returns
	    ----------
	    float
	    	Mse of the model predictions with respect to the experimental data
	    """

		# Estimating model predictions on the benchmarking samples
		self.model.eval()
		X, C = benchmarks['X'], benchmarks['C']
		perf, u, s, c_hat, c_pr = self.benchmark(X, C)
		self.model.train()

		# Figure config
		fig  = plt.figure()
		gs   = fig.add_gridspec(4,6)
		ax_ex = [fig.add_subplot(gs[n, :]) for n in range(2)] + [fig.add_subplot(gs[2, :3])]
		ax_dd_mse, ax_dd = fig.add_subplot(gs[2, 3]), fig.add_subplot(gs[2, 4:])
		ax_mse, ax_kol  = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
		ax_cacc, ax_cce = fig.add_subplot(gs[3, 2]), fig.add_subplot(gs[3, 3])
		ax_loss = fig.add_subplot(gs[3, 4:]) 

		# Plotting a few examples
		for n in range(len(ax_ex)):
			plot_predictions(X[n], u[n], s[n], C[n], c_hat[n], c_pr[n],  ax_ex[n])
			ax_ex[n].set_xlim([0, X.shape[1]])
		ax_ex[n].set_xlim([0, 2400])


		# Estimating responses to data's paradigm
		self.model.eval()
		u = self.run(torch.tensor(dd['x']).to(self.dev).float())[0].detach().cpu().numpy()[..., 0]
		self.model.train()		
		mod_avg, mod_sem = u.mean(0)[dd['t']], u.std(0)[dd['t']] / np.sqrt(u.shape[0])
		coin_avg, coin_sem = dd['u'].mean(0), dd['u'].std(0) / np.sqrt(dd['u'].shape[0])

		ax_dd.fill_between(dd['t'], coin_avg-coin_sem, coin_avg+coin_sem, color='tab:green', alpha=0.2)
		ax_dd.fill_between(dd['t'], mod_avg-mod_sem, mod_avg+mod_sem, color='k', alpha=0.2)
		ax_dd.plot(dd['f'][0,:,0].detach().cpu().numpy(), color='tab:blue', label='field')
		ax_dd.plot(dd['t'], mod_avg, color='k', label='GRU')
		ax_dd.plot(dd['t'], coin_avg, color='tab:green', label='coin')
		ax_dd.plot(dd['t'], dd['y'].detach().cpu().numpy(), color='tab:orange', label='participant')
		ax_dd.set_xlabel('trial number')
		ax_dd.set_ylabel('field / estimation')

		# Plotting data fit
		mod_mse  = ((u[:, dd['t']] - dd['y'].cpu().detach().numpy())**2).mean(1)
		coin_mse = ((dd['u'] - dd['y'].cpu().detach().numpy())**2).mean(1)
		mod_mse_avg, mod_mse_sem = mod_mse.mean(), mod_mse.std() / np.sqrt(mod_mse.shape[0])
		coin_mse_avg, coin_mse_sem = coin_mse.mean(), coin_mse.std() / np.sqrt(coin_mse.shape[0])
		mse_r = mod_mse_avg / mod_mse_sem

		colours = ['tab:green', 'tab:gray'] 
		vals, errs = [coin_mse_avg, mod_mse_avg], [coin_mse_sem, mod_mse_sem]
		ax_dd_mse.bar([0, 1], vals, yerr=errs, color=colours)
		ax_dd_mse.set_xticks([0, 1])
		ax_dd_mse.set_xticklabels(['COIN', 'GRU'])
		ax_dd_mse.set_ylabel('mse')

		# Plotting Performance
		perf = {**{'gru': perf}, **benchmarks['perf']}
		plotvec = dict()
		for key in ['mse', 'kol', 'ce']:
			plotvec[key] = dict()
			for val in ['avg', 'sem']:
				plotvec[key][val] =  [perf[mod][key][val] for mod in ['LI', 'coin', 'gru']]

		for key in ['ct_ac', 'ct_ce']:
			plotvec[key] = dict()
			for val in ['avg', 'sem']:
				plotvec[key][val] =  [perf[mod][key][val] for mod in ['coin', 'gru']]

		colours = ['tab:purple', 'tab:green', 'tab:gray'] 
		ax_mse.bar([0, 1, 2], plotvec['mse']['avg'], yerr=plotvec['mse']['sem'], color=colours)
		ax_mse.set_xticks([0, 1, 2])
		ax_mse.set_xticklabels(['Leak-Int', 'COIN', 'GRU'])
		ax_mse.set_ylabel('mse')
		ax_kol.bar([0, 1, 2], plotvec['kol']['avg'], yerr=plotvec['kol']['sem'], color=colours)
		ax_kol.set_xticks([0, 1, 2])
		ax_kol.set_xticklabels(['Leak-Int', 'COIN', 'GRU'])
		ax_kol.set_ylabel('KS-stat')
		
		colours = ['tab:green', 'tab:gray'] 
		ax_cacc.bar([0, 1], plotvec['ct_ac']['avg'], yerr=plotvec['ct_ac']['sem'], color=colours)
		ax_cacc.set_xticks([0, 1])
		ax_cacc.set_xticklabels(['COIN', 'GRU'])
		ax_cacc.set_ylabel('ctx accuracy')
		ax_cce.bar( [0, 1], plotvec['ct_ce']['avg'],  yerr=plotvec['ct_ce']['sem'],  color=colours)
		ax_cce.set_xticks([0, 1])
		ax_cce.set_xticklabels(['COIN', 'GRU'])
		ax_cce.set_ylabel('ctx CE')

		# Plotting loss evolution
		t0 = 0
		colours = ['k', 'tab:gray']
		for e in range(len(lossHistory)):
			trials = batch_res * np.arange(t0, t0 + len(lossHistory[e]))
			ax_loss.plot(trials, np.array(lossHistory[e]), color=colours[(epoch0+e)%2])
			t0 += len(lossHistory[e])
		ymin = 0.2 * np.floor(5 * np.array(lossHistory).min()) - 0.2
		ymax = 0.2 * min(25, 5 * np.ceil(np.array(lossHistory).max())) + 0.1
		ax_loss.set_ylim([ymin, ymax])
		ax_loss.set_xlabel('batch number')
		ax_loss.set_ylabel('loss')

		# Marking last 5 minima
		lharray = np.array(lossHistory)
		ix = np.unravel_index(lharray.reshape(-1).argsort()[:5], lharray.shape)
		for e, b, n in zip(*ix, range(len(ix[0]))):
			trial = batch_res * (sum([len(lossHistory[e0]) for e0 in range(e)]) + b)
			c0 = np.array(matplotlib.colors.to_rgb('tab:red'))
			c1 = np.array(matplotlib.colors.to_rgb('tab:blue'))
			c1 = np.array([255, 239, 69]) / 255
			c0 = np.array([255, 0, 0]) / 255
			cTrial = matplotlib.colors.to_hex((1-(n/(len(ix[0])-1)))*c0 + (n/(len(ix[0])-1))*c1)
			ax_loss.plot(trial, lossHistory[e][b], 'v', color=cTrial)

		# Saving
		fig.set_size_inches(23, 13)
		fig.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95,wspace=0.3,hspace=0.3)
		plt.savefig(f'./logs/{savename.split(".")[0]}-e{len(lossHistory):03}.png')
		plt.cla()
		plt.close(fig)

		return mse_r

	# COIN Experiments
	def run_experiment_coin(self, experiment, N=20, axs=None):
		""" Computes the prediction of the wrapped nn.Module under the fields of an experiment from
			the COIN paper
	    
	    Parameters
	    ----------
	    experiment : str
	    	Reference to the experiment; see coin.generate_field_sequence() for details.
	    N          : int (optional)
	    	Number of runs
		axs     : (optional) an instance of plt.subplots()
			plt axes where to plot the results

	    Returns
	    ----------
	    u  : dict() (optional)
	    	A dictionary encoding the predictions across the experiment conditions (keys); same 
	    	structure as each of the sub-dicts of U produced by coin.initialise_experiments_U
		x0  : np.array
			field values for one of the runs; dim 0 is set to 1, dim 1 runs across time points, 
			dim 2 is set to one. Useful for plotting.
	    """

		self.model.eval()

		match experiment:
			case 'consistency':
				x = generate_field_sequence(experiment, self.sigma_r, 1, pStay=0.5)
				nanix = torch.where(torch.isnan(x[0]))[0]
				triplets = [ix for ix in nanix if ix+2 in nanix]
				x0 = x[0, [t+1 for t in triplets], 0].detach().cpu()

				pStayList, u = [0.1, 0.5, 0.9], dict()
				for p in pStayList:
					x = generate_field_sequence(experiment, self.sigma_r, N, pStay=p).to(self.dev)
					u0 = self.run(x)[0].detach().cpu()
					u[p] = torch.stack([u0[:, t+2, 0]-u0[:,t, 0] for t in triplets], axis=1).numpy()

			case 'interference':
				nPlusList, u = [0, 13, 41, 112, 230, 369], dict()
				for nPlus in nPlusList:
					x = generate_field_sequence(experiment, self.sigma_r, N, Np=nPlus).to(self.dev)
					u[nPlus] = self.run(x)[0][:, (160+nPlus):, 0].detach().cpu().numpy()
				x0 = x[0, 160+nPlus:, 0].detach().cpu().numpy()

			case 'savings':
				x = generate_field_sequence(experiment, self.sigma_r, N).to(self.dev)
				u0 = self.run(x)[0].detach().cpu().numpy()
				t0, t1, dur = 60, 60+125+15+50+50+60, 125
				u = {'first': u0[:, t0:(t0+dur), 0], 'second': u0[:, t1:(t1+dur), 0]}
				x0 = x[0, t0:(t0+dur), 0].detach().cpu().numpy()

			case 'spontaneous' | 'evoked':
				x = generate_field_sequence(experiment, self.sigma_r, N).to(self.dev)
				u = {'data': self.run(x)[0][..., 0].detach().cpu().numpy()}
				x0 = x[0, :, 0].detach().cpu().numpy()

		if axs is not None:
			coin.plot_experiment(x0, u, experiment, axs=axs)

		return(u, x0)

	def all_coin_experiments(self, N=20):
		""" Computes the predictions of the wrapped nn.Module under the fields for all the experiments 
			of the COIN paper that do not involve cue emissions.

	    Parameters
	    ----------
	    N   : int (optional)
	    	Number of runs

	    Returns
	    ----------
	    U   : dict
			Dictionary with np.arrays encoding the predictions hat{y}_t for the fields of each
			experiment (keys); dim 0 runs across runs, dim 1 across time points, dim 2 set to one.
		x0  : np.array
			field values for one of the runs; dim 0 is set to 1, dim 1 runs across time points, 
			dim 2 is set to one. Useful for plotting.
	    """

		experiments = ['spontaneous', 'evoked', 'savings', 'interference', 'consistency']
		U, X0 = dict(), dict()
		for exp in experiments:
			U[exp], X0[exp] = self.run_experiment_coin(exp, N)

		return U, X0

	def summary_experiments(self, savefig=None, N=20, cmap="Greys", axs=None, eng=None):
		""" Prints a figure withe the predictions of the wrapped nn.Module for all the experiments of 
			the COIN paper.

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
	    """

		U, X0 = self.all_coin_experiments(N)

		if axs is None:
			fig, axs = plt.subplots(1, 5)
		else:
			fig = None

		for experiment, ax in zip(U, axs):
			d[experiment] = coin.plot_experiment(U[experiment], experiment, ax, cmap)

		if savefig is not None and fig is not None:
			fig.subplots_adjust(left=0.05,right=0.98,bottom=0.17,top=0.9,wspace=0.3,hspace=0.3)
			fig.set_size_inches(20, 3)
			plt.savefig(f'{savefig}.png')




def measure_KS_stat(x, u, s):
	""" Measures the calibration (K-S statistic) of a prediction on a set of observations 

	    Parameters
	    ----------
	    x : np.array
	    	Input array enconding sequences of observations; three dimensions: dim0 runs across the
	    	batch, dim1 runs across timepoints (i.e., trials in the COIN jargon), dim2 set to one.
		u  : np.array
			Mean of the predictions; shifted (i.e., u[:, t] are the predictions for t). Same 
			dimensions as x.
		s  : np.array
			Standard deviation of the predictions; same shape and shifting as u.

	    Returns
	    ----------
	    KS : np.array
	    	One dimensional array with a scalar per sample in the batch. 
	"""

	F    = np.linspace(0, 1, 1000)
	cump = (0.5 * (1 + scipy.special.erf((x - u) / (np.sqrt(2) * s))))
	N    = (~np.isnan(cump)).sum(1)
	KS   = abs(np.array([(cump <= f).sum(1) / N for f in F]) - F[:, None, None]).max((0, 2))
	
	return KS


def plot_predictions(x, u, s, c=None, c_hat=None, c_pr=None, ax=None):
	""" Auxiliary function that plots a single line of observations/fields, an average prediction,
		and its associated uncertainty. 

	    Parameters
	    ----------
	    x  : torch.tensor
	    	A one-dimensional array encoding the line of observations/fields 
		u  : torch.tensor (optional)
			A one-dimensional array encoding the average prediction
		s  : torch.tensor (optional)
			A one-dimensional array encoding the uncertainty on the prediction (e.g., std)
		c  : torch.tensor (optional)
	    	A one-dimensional array encoding the contexts corresponding to the fields; when 
	    	specified, observations are colour-coded corresponding to their context
		axs  : (optional) an instance of plt.subplots()
			plt axes where to plot the prediction
	"""

	if ax is None:
		ax = plt.axes()

	field, mu, sigma, context = numpyFlatten(x), numpyFlatten(u), numpyFlatten(s), numpyFlatten(c)
	if c_hat is not None:
		c_pred, c_prob = numpyFlatten(c_hat).astype(int), numpyFlatten(c_pr)

	ax.plot(range(len(field)), mu, color='k')
	ax.fill_between(range(len(field)), mu-sigma, mu+sigma, color='k', alpha=0.2)
	ax.set_xlabel('trial number')
	ax.set_ylabel('field / output')

	bottom = 0.1 * np.floor(min(field.min(), (mu-sigma).min()) * 10)
	top    = 0.1 * np.ceil(max(field.max(), (mu+sigma).max()) * 10)

	if context is None:
		ax.plot(range(len(field)), field, color='tab:blue')
	else:
		nColCycles = int(np.ceil((max(max(context), max(c_pred)) + 1) / 10))
		colours = list(matplotlib.colors.TABLEAU_COLORS.values()) * nColCycles
		time = np.arange(len(field), dtype=int)

		T = [0] + [t for t in range(1, len(field)) if (c[t]-c[t-1])!=0] + [len(field)-1]
		for t0, t1 in zip(T[:-1], T[1:]):
			ax.plot(time[t0:(t1+1)], field[t0:(t1+1)], color = colours[context[t0]])
			if c_hat is not None and c_pred[t0] != -1:
				ax.plot(time[t0:(t1+1)], [bottom - 0.02*abs(bottom)] * (t1-t0+1) , color=colours[context[t0]])

		if c_hat is not None and c_pred[t0] != -1:
			for t in time[:-1]:
				ax.plot([time[t], time[t+1]], [bottom]*2, '-', color=colours[c_pred[t]], alpha=c_prob[t])
		
		ax.set_ylim([bottom - 0.1 * abs(bottom), top + 0.1 * abs(top)])
		ax.set_xlim([0, len(field)])


def generate_field_sequence(experiment, noise=0.03, batch_size=1, **kwargs):
	""" Wrapper for coin.generate_field_sequence that returns a torch.tensor

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
	    x  : torch.tensor
	    	sequence of observations; dim 0 runs across batches, dim 1 across time points, dim 2 
	    	is set to one for compatibility with pytorch
	"""

	x = torch.tensor(coin.generate_field_sequence(experiment, noise, batch_size, **kwargs)).float()
	return(x)


def numpyFlatten(z):
	""" Transforms an n-dimensional torch.tensor into an irreducible (i.e., no dimensions of dim 1)
		numpy array.

	    Parameters
	    ----------
	    a : torch.tensor
	    	can be on any device and have or not stored gradients
	   
	    Returns
	    -------
		z : np.array

	"""


	if type(z) == torch.Tensor:
		if z.requires_grad:
			z = z.detach()
		z = z.cpu().numpy()
	if len(z.shape) == 3:
		zflat = z[0, :, 0]
	elif len(z.shape) == 2:
		if z.shape[0] >= z.shape[1]:
			zflat = z[:, 0]
		else:
			zflat = z[0, :]
	elif len(z.shape) == 1:
		zflat = z

	return zflat



def find_latest(modname, pars=None):
	""" Estimates the predictions of a model across a set of subjects for all the
		experiments from the COIN paper that do not involve cue emissions. 

	    Parameters
	    ----------
	    modkey   : str
	    	Filename specified to save the statedict during training.
	   
	    Returns
	    -------
	    modname  : str
 			Path to the statedict of the latest	(highest epoch) instance of the specified model in
 			./models/.
	"""

	if pars is not None:
		modname = f'pars-{pars}_{modname}'
	if modname.split('.')[-1] == '.pt':
		modname = '.'.join(modname.split('.')[:-1])
	if modname[0] != '/' and modname[0] != './':
		modname = './models/' + modname

	return sorted(glob.glob(f'{modname}_e*'))[-1].split('/')[-1]



def summary_stats_fit(modkey, subs=None):
	""" Prints a summary figure with the average prediction of all the models fitted to each of the 
		subjects for all the experiments of the COIN paper that do not involve cue emissions. 

	    Parameters
	    ----------
	    modkey : str
	    	It assumes models were saved as f'pars-{sub}_{modkey}' where sub is the parset of 
	    	each of the subjects (e.g., 'S1'). It will use modkey to find_latest to locate the 
	    	latest (highest epoch) instance of the model corresponding to each subject in ./models.

	    subs   : str or list of str (optional)
	    	Each string identifies a subject (e.g., 'S1') or a set of subjects (e.g., 'S'). If not
	    	specified it uses all 'S' and 'E' subjects.
	"""


	if subs is None:
		subs = list(coin.load_sub_pars(['S', 'E']).keys())

	fig, axs = plt.subplots(4, len(subs))

	for n, sub in enumerate(subs):

		print(f'Plotting fits for subject {sub}')
		# Recovering GM benchmarks
		gm = coin.CRFGenerativeModel(sub)
		benchmarks = gm.benchmark(n_trials=5000)

		# Recovering model
		modpath = sorted(glob.glob(f'./models/pars-{sub}_{modkey}*'))[-1]
		m = Model()
		m.load_weights(modpath)

		# Estimating model predictions on the benchmarking samples
		m.model.eval()
		X, C = benchmarks['X'], benchmarks['C']
		perf, u, s, c_hat, c_pr = m.benchmark(X, C)
		m.model.train()

		# Preparing plotbars
		perf = {**{'gru': perf}, **benchmarks['perf']}
		plotvec = dict()
		for key in ['mse', 'kol', 'ce']:
			plotvec[key] = dict()
			for val in ['avg', 'sem']:
				plotvec[key][val] = [perf[mod][key][val] for mod in ['LI', 'coin', 'gru']]

		for key in ['ct_ac', 'ct_pr']:
			plotvec[key] = dict()
			for val in ['avg', 'sem']:
				plotvec[key][val] = [perf[mod][key][val] for mod in ['coin', 'gru']]

		# Plotting
		colours = ['tab:purple', 'tab:green', 'tab:gray'] 
		axs[0,n].bar([0, 1, 2], plotvec['mse']['avg'], yerr=plotvec['mse']['sem'], color=colours)
		axs[0,n].set_xticks([0, 1, 2])
		axs[0,n].set_xticklabels(['Leak-Int', 'COIN', 'GRU'])
		axs[0,n].set_ylabel('mse')
		axs[1,n].bar([0, 1, 2], plotvec['kol']['avg'], yerr=plotvec['kol']['sem'], color=colours)
		axs[1,n].set_xticks([0, 1, 2])
		axs[1,n].set_xticklabels(['Leak-Int', 'COIN', 'GRU'])
		axs[1,n].set_ylabel('KS-stat')
		
		colours = ['tab:green', 'tab:gray'] 
		axs[2,n].bar([0, 1], plotvec['ct_ac']['avg'], yerr=plotvec['ct_ac']['sem'], color=colours)
		axs[2,n].set_xticks([0, 1])
		axs[2,n].set_xticklabels(['COIN', 'GRU'])
		axs[2,n].set_ylabel('ctx accuracy')
		axs[3,n].bar( [0, 1], plotvec['ct_pr']['avg'],  yerr=plotvec['ct_pr']['sem'],  color=colours)
		axs[3,n].set_xticks([0, 1])
		axs[3,n].set_xticklabels(['COIN', 'GRU'])
		axs[3,n].set_ylabel('ctx CE')

		axs[0,n].set_title(f'{sub} (epoch {modpath.split("-")[-1].split(".")[0][1:]})')

	txtspec = dict(ha='left', va='center', fontsize=12, color='tab:blue')
	plt.text(-0.2, 1.3, modkey, rotation=0,  transform=axs[0,0].transAxes, **txtspec)
	fig.subplots_adjust(left=0.02,right=0.98,bottom=0.05,top=0.9,wspace=0.45,hspace=0.4)
	fig.set_size_inches(40, 8)
	plt.savefig(f'summaryperformance-{modkey}.png')



def estimate_subject_fits(modkey, subs=None):
	""" Estimates the predictions of a model across a set of subjects for all the
		experiments from the COIN paper that do not involve cue emissions. 

	    Parameters
	    ----------
	    modkey : str
	    	It assumes models were saved as f'pars-{sub}_{modkey}' where sub is the parset of 
	    	each of the subjects (e.g., 'S1'). It will use modkey to find_latest to locate the 
	    	latest (highest epoch) instance of the model corresponding to each subject in ./models.

	    subs   : str or list of str (optional)
	    	Each string identifies a subject (e.g., 'S1') or a set of subjects (e.g., 'S'). If not
	    	specified it uses all 'S' and 'E' subjects.

	    Returns
	    -------
	    U : dict()
	    	A dictionary encoding the predictions across subjects and experiments; same structure 
	    	as produced by coin.initialise_experiments_U
	"""

	if subs is None:
		subs = ['S', 'E']

	subs = list(coin.load_sub_pars(subs).keys())

	U = coin.initialise_experiments_U(len(subs))

	for s, sub in enumerate(subs):
		print(sub)
		m = Model()
		#m.load_weights(find_latest(f'pars-{sub}_pretrained-baseline_{modkey}'))
		m.load_weights(find_latest(f'modkey'))
		Usub, X0 = m.all_coin_experiments()
		for key in Usub:
			if type(Usub[key]) is dict:
				for subkey in U[key]:
					U[key][subkey][s] = Usub[key][subkey].mean(0)
			else:
				U[key]['data'][s] = Usub[key].mean(0)

	return U


