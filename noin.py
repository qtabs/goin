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
DEV='cuda'
EPS=torch.tensor(1E-10)

# This is an unfinished library; most functions are working correctly, except for the computation of 
# predictions in paradigms that involve channel trials: these work but are still not differentiable 
# (solution probably involves detaching an intermediate variable during the computation of the fictious
# observations, but I haven't yet figure this out). The library is not documented, but all the methods
# and functions are fully documented in the version of the library that considers a RNNs with a single
# Gaussian as output (goin.py). Most documentation from goin.py should be trivially applicable to this 
# library.

class GruRNN(nn.Module):

	def __init__(self, n_hidden=64, n_layers=1, n_out=11, batch_size=None, train_h0=False, dev=DEV):
		
		super(GruRNN, self).__init__()
		
		# Init pars:
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.n_out    = n_out
		self.dev      = dev

		# Harcoded pars:
		dropout      = 0.0
		dim_in       = 1

		# Networks
		self.gru = nn.GRU(dim_in, n_hidden, n_layers, batch_first=True, dropout=0, device=dev)
		self.out_mu    = nn.Linear(n_hidden, n_out, bias=True, device=dev)
		self.out_sigma = nn.Linear(n_hidden, n_out, bias=True, device=dev)
		self.out_lamb  = nn.Linear(n_hidden, n_out, bias=True, device=dev)

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

		out_mu    = self.out_mu(x)
		out_sigma = self.out_sigma(x)
		out_lamb  = self.out_lamb(x)

		return out_mu, out_sigma, out_lamb


class Model():

	def __init__(self, modelpath=None, sigma_r=0.1, dev=DEV, **pars):

		self.dev      = dev
		self.sigma_r  = sigma_r
		self.model    = GruRNN(**pars, dev=self.dev)

		if modelpath is not None:
			self.load_weights(modelpath)
		
	def load_weights(self, modelpath):
		if '.pt' not in modelpath:
			modelpath += '.pt'
		if modelpath[0] != '/':
			if './models' not in modelpath:
				modelpath = './models/' + modelpath
		self.model.load_state_dict(torch.load(modelpath))
		self.model.gru.flatten_parameters() 

	def save_weights(self, modelpath, epoch=None):
		
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
		mu, out_sigma, out_lamb = self.model(trg)
		sigma = torch.log(1 + torch.exp(out_sigma))

		lambmask = torch.ones(out_lamb.shape).to(self.dev).float()
		for t in range(t0+1, min(out_lamb.shape[1:3])):
			lambmask[:, t-(t0+1), t+1:] = 1E-30

		lamb = torch.log((1 + torch.exp(out_lamb)) * lambmask)
		lamb = lamb - torch.logsumexp(lamb, dim=2, keepdim=True)
		return (mu, sigma, lamb)

	def run(self, x):

		n_batches, seq_len, dim_in = x.shape
		n_out = self.model.n_out

		if isinstance(self.model, GruRNN):
			self.model.init_hidden(n_batches)

		u = (torch.zeros((n_batches, seq_len, n_out), requires_grad=False) * torch.nan).to(self.dev)
		s = (torch.zeros((n_batches, seq_len, n_out), requires_grad=False) * torch.nan).to(self.dev)
		l = (torch.zeros((n_batches, seq_len, n_out), requires_grad=False) * torch.nan).to(self.dev)
		l[:, 0, :] = -30
		l[:, 0, 0] = 0

		if torch.any(torch.isnan(x)) and isinstance(self.model, GruRNN):
			for b in range(n_batches):
				chantrials = torch.where(torch.any(torch.isnan(x[b, :, :]), 1))[0]
				t0 = 0
				for t1 in chantrials:
					u[b, (t0+1):(t1+1), :], s[b, (t0+1):(t1+1), :], l[b, (t0+1):(t1+1), :] = self.call(x[[b], t0:t1, :])
					x[b, t1, :] =(torch.exp(l[b, t1, :]) * u[b, t1, :]).sum() + self.sigma_r * torch.randn(dim_in).to(self.dev)
					t0 = copy.deepcopy(t1)
		else:
			u[:, 1:, :], s[:, 1:, :], l[:, 1:, :] = self.call(x[:, :-1, :])

		return u, s, l

	def predict(self, x):

		u, s, l = self.run(x)
		lamb = torch.exp(l)

		mu    = (lamb * u).sum(2)
		sigma = ((lamb * s**2).sum(2) + (lamb * u**2).sum(2) - mu**2)**0.5

		return mu[:, :, None], sigma[:, :, None], l 

	def benchmark(self, x, c):

		nb, nt = x.shape[0], x.shape[1]
		u, s = np.zeros((nb, nt)), np.zeros((nb, nt)) 
		l    = np.zeros((nb, nt, self.model.n_out))
		llk  = np.zeros(nb)

		bunches = [(64*n, min(64*(n+1), x.shape[0])) for  n in range(int(np.ceil(x.shape[0]/64)))]
		for b in bunches:
			z = torch.tensor(x[b[0]:b[1]], dtype=torch.float, requires_grad=False).to(self.dev)
			with torch.no_grad():
				bu, bs, bl = self.run(z)

			bu,bs,bl = bu.cpu().detach().numpy(),bs.cpu().detach().numpy(),bl.cpu().detach().numpy()
			lamb = np.exp(bl)

			u[b[0]:b[1], :] = (lamb * bu).sum(2)
			s[b[0]:b[1], :] = ((lamb*bs**2).sum(2) + (lamb*bu**2).sum(2) - (lamb*bu).sum(2)**2)**0.5
			l[b[0]:b[1], : ,:] = bl

			Z = -1/2 * np.log(2 * np.pi) - np.log(np.maximum(bs, EPS.numpy()))
			llk[b[0]:b[1]]  = np.nanmean(scipy.special.logsumexp(Z + bl - (bu-x[b])**2/(2*bs), 2), 1)

		# Observations
		mse = (u[..., None] - x)**2
		KS  = measure_KS_stat(x, u[..., None], s[..., None])
	
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
		perf = {'mse': {}, 'kol': {}, 'ce': {}, 'ct_ac': {}, 'ct_ce': {}}

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

		# Hardcoded paramters
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

		y, q, c = gm.generate_batch(n_trials, batch_size)
		x = torch.tensor(y, dtype=torch.float, requires_grad=False).to(self.dev)
		u, s, l = self.run(x)

		# Loss over the predictions for the observations
		loss_obs = -self._mixture_log_prob_(u, s, l, x, t=range(1, x.shape[1]))
		
		# Loss over the predictions for the context
		if ctx:
			cflat = c[...,0].reshape(-1)
			lflat = l.reshape(-1, l.shape[2])
			logp_ctx = [torch.nansum(lflat[cflat==ctx, ctx]) for ctx in range(l.shape[2])]
			loss_ctx = -torch.stack(logp_ctx).sum() / len(cflat)
		else:
			loss_ctx = torch.tensor(0)

		# Loss over the experimental data fit
		# !!! ToDo: dd loss backwards() fails when computing the mean of the ficticious 
		#           observations as mu = sum(l * u), but not when computing them as mu = u.
		#           Solution needed.
		if dd is not None:
			noise      = gm.si_r * torch.randn(dd['f'].shape).to(self.dev).float()
			u2, s2, l2 = self.run(dd['f'] + noise)
			loss_dd    = -self._mixture_log_prob_(u2, s2, l2, dd['f'], t=dd['t'])
		else:
			loss_dd = torch.tensor(0)

		return loss_obs, loss_ctx, loss_dd

	def _mixture_log_prob_(self, u, s, l, x, t=None):

		if t == None:
			t = range(x.shape[1])

		Z = torch.tensor(-1/2 * np.log(2 * np.pi)) - torch.log(torch.maximum(s[:, t, :], EPS))
		d = (u[:, t, :] - x[:, t, :]) ** 2 / (2 * s[:, t, :])
		logp = torch.logsumexp(Z + l[:, t, :] - d, 2)

		return torch.mean(logp)

	def set_optim(self, lr, train_sched, optimise=None, freeze=None):

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
		batch_res    = 1    # Store and report loss every batch_res batches
		n_trials     = 5000
		batch_size   = 64
		return n_trials, batch_size, batch_res
	
	def loss_log(self, logpath, batch, batch_res, n_batches, tt, lr, loss, losses = (None,None,None)):
		
		loss_obs, loss_ctx, loss_dd = losses

		loss_np = loss.detach().item()
		sprint  = f'Batch {batch+1:>3}/{n_batches}; Time = {time.time()-tt:.1f}s; '
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
			fcols = C[n,:].reshape(-1)
			plot_predictions(X[n], u[n], s[n], C[n], c_hat[n], c_pr[n],  ax_ex[n])
			ax_ex[n].set_xlim([0, X.shape[1]])

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
		u = self.predict(torch.tensor(dd['x']).to(self.dev).float())[0].detach().cpu().numpy()[..., 0]
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
					u0 = self.predict(x)[0].detach().cpu()
					u[p] = torch.stack([u0[:, t+2, 0]-u0[:,t, 0] for t in triplets], axis=1).numpy()

			case 'interference':
				nPlusList, u = [0, 13, 41, 112, 230, 369], dict()
				for nPlus in nPlusList:
					x = generate_field_sequence(experiment, self.sigma_r, N, Np=nPlus).to(self.dev)
					u[nPlus] = self.predict(x)[0][:, (160+nPlus):, 0].detach().cpu().numpy()
				x0 = x[0, 160+nPlus:, 0].detach().cpu().numpy()

			case 'savings':
				x = generate_field_sequence(experiment, self.sigma_r, N).to(self.dev)
				u0 = self.predict(x)[0].detach().cpu().numpy()
				t0, t1, dur = 60, 60+125+15+50+50+60, 125
				u = {'first': u0[:, t0:(t0+dur), 0], 'second': u0[:, t1:(t1+dur), 0]}
				x0 = x[0, t0:(t0+dur), 0].detach().cpu().numpy()

			case 'spontaneous' | 'evoked':
				x = generate_field_sequence(experiment, self.sigma_r, N).to(self.dev)
				u = {'data': self.predict(x)[0][..., 0].detach().cpu().numpy()}
				x0 = x[0, :, 0].detach().cpu().numpy()

		if axs is not None:
			coin.plot_experiment(u, experiment, axs=axs)

		return(u, x0)

	def all_coin_experiments(self, N=20):

		experiments = ['spontaneous', 'evoked', 'savings', 'interference', 'consistency']
		U, X0 = dict(), dict()
		for exp in experiments:
			U[exp], X0[exp] = self.run_experiment_coin(exp, N)

		return U, X0

	def summary_experiments(self, savefig=None, N=20, cmap="Greys", axs=None, eng=None):
		
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
	F    = np.linspace(0, 1, 1000)
	cump = (0.5 * (1 + scipy.special.erf((x - u) / (np.sqrt(2) * s))))
	N    = (~np.isnan(cump)).sum(1)
	KS   = abs(np.array([(cump <= f).sum(1) / N for f in F]) - F[:, None, None]).max((0, 2))
	return KS


def plot_predictions(x, u, s, c=None, c_hat=None, c_pr=None, ax=[]):

	if ax == []:
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
				ax.plot(time[t0:(t1+1)], [1.02 * bottom] * (t1-t0+1) , color=colours[context[t0]])

		if c_hat is not None and c_pred[t0] != -1:
			for t in time[:-1]:
				ax.plot([time[t], time[t+1]], [bottom]*2, '-', color=colours[c_pred[t]], alpha=c_prob[t])
		
		ax.set_ylim([1.1*bottom, 1.1*top])


def generate_field_sequence(experiment, noise=0.03, batch_size=1, **kwargs):
	x = torch.tensor(coin.generate_field_sequence(experiment, noise, batch_size, **kwargs)).float()
	return(x)


def numpyFlatten(z):

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


def summary_stats_fit(modkey, subs=None):

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


def find_latest(modname, pars=None):
	if pars is not None:
		modname = f'pars-{pars}_{modname}'
	if modname.split('.')[-1] == '.pt':
		modname = '.'.join(modname.split('.')[:-1])
	if modname[0] != '/' and modname[0] != './':
		modname = './models/' + modname

	return sorted(glob.glob(f'{modname}_e*'))[-1].split('/')[-1]


def estimate_subject_fits(modkey, subs=None):

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


