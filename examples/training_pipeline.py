import sys
import json
import os
sys.path.append('../')
import goin
import numpy as np


PROGRESS_FILE = 'training_progress.txt'


def load_completed_combinations():
	"""Load list of completed parameter combination indices."""
	if not os.path.exists(PROGRESS_FILE):
		return set()
	with open(PROGRESS_FILE, 'r') as f:
		return set(int(line.strip()) for line in f if line.strip())


def mark_combination_complete(ix):
	"""Mark a parameter combination as completed."""
	with open(PROGRESS_FILE, 'a') as f:
		f.write(f"{ix}\n")


def generate_parname(pvals, baseline):
	"""Generate parameter name from varied parameters only."""
	varied_pars = {k: v for k, v in pvals.items() if k not in baseline or baseline[k] != v}

	if not varied_pars:
		return "baseline"

	name_parts = []
	for key in sorted(varied_pars.keys()):
		value = varied_pars[key]
		# Format with appropriate precision
		if isinstance(value, float):
			if abs(value) < 0.1:
				formatted = f"{value:.3f}"
			elif abs(value) < 1.0:
				formatted = f"{value:.2f}"
			else:
				formatted = f"{value:.1f}"
		else:
			formatted = str(value)
		name_parts.append(f"{key}-{formatted}".replace('.', 'p'))

	return "_".join(name_parts)


def load_parameter_combinations(config_path):
	"""Load parameter combinations from JSON config file."""
	with open(config_path, 'r') as f:
		config = json.load(f)

	baseline = config['baseline']
	pvals_list = config['parameter_combinations']

	return baseline, pvals_list


def run_training(pvals, baseline, device='cpu', n_hidden=64, n_batches=500,
				 n_epochs_mse=40, n_epochs_ctx=10, n_epochs_ora=45, oracle=0.2, lr=0.001):
	"""Run training for a specific parameter combination."""

	# Merge baseline with varied parameters
	pars = baseline.copy()
	pars.update(pvals)

	# Generate parameter name
	parname = generate_parname(pvals, baseline)
	parset = {'name': parname, 'pars': pars}

	print(f"Training with parameters: {parname}")
	# print(f"Full parameter set: {pars}")

	# Training oracled model
	m = goin.Model(n_hidden=n_hidden, dev=device)
	m.train(parset, f'{parname}_{n_hidden}_ora', oracle=oracle, lr=lr, train_sched=(n_epochs_ora, n_batches))

	# Training non-oracled model
	m = goin.Model(n_hidden=n_hidden, dev=device)
	m.train(parset, f'{parname}_{n_hidden}_mse', oracle=0.0, lr=lr, train_sched=(n_epochs_mse, n_batches))
	m.train_ctx(parset, f'{parname}_{n_hidden}_ctx', lr=lr, train_sched=(n_epochs_ctx, n_batches))


def print_header(ix, pvals_list):
	"""Print formatted header for training progress."""
	total_digits = len(str(len(pvals_list)))
	msg = f' Parameter combination {ix+1:0{total_digits}d}/{len(pvals_list)} '
	border = '┌' + '─' * (len(msg)) + '┐'
	bottom = '└' + '─' * (len(msg)) + '┘'
	print(f'\n\n{border}\n│{msg}│\n{bottom}\n')


if __name__ == '__main__':
	# Check CUDA availability
	import torch
	USE_CUDA = torch.cuda.is_available()
	DEVICE = 'cuda' if USE_CUDA else 'cpu'
	print(f"Using device: {DEVICE}")

	# Determine config file path
	if len(sys.argv) > 2:
		config_path = sys.argv[2]
	else:
		config_path = os.path.join(os.path.dirname(__file__), 'training_config.json')

	# Load parameter combinations
	baseline, pvals_list = load_parameter_combinations(config_path)
	print(f"Loaded {len(pvals_list)} parameter combinations from {config_path}")

	# Load completed combinations
	completed = load_completed_combinations()
	if completed:
		print(f"Found {len(completed)} already completed combinations")

	# Run specific combination or all
	if len(sys.argv) > 1 and sys.argv[1].isdigit():
		ix = int(sys.argv[1])
		if ix in completed:
			print(f"Combination {ix} already completed, skipping")
		else:
			print_header(ix, pvals_list)
			run_training(pvals_list[ix], baseline=baseline, device=DEVICE)
			mark_combination_complete(ix)
	else:
		for ix, pvals in enumerate(pvals_list):
			if ix in completed:
				print(f"Skipping combination {ix+1}/{len(pvals_list)} (already completed)")
				continue
			print_header(ix, pvals_list)
			run_training(pvals, baseline=baseline, device=DEVICE)
			mark_combination_complete(ix)
