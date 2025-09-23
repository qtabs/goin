import sys
sys.path.append('../')
import goin
import numpy as np

def run_training(pvals):
	
	# Training hyperparameters
	n_batches    = 500
	n_epochs_mse = 40
	n_epochs_ctx = 10
	n_epochs_ora = 45
	oracle   = 0.2
	lr       = 0.001
	n_hidden = 64

	# Baseline set of parameters
	pars = {'alpha_t' : 0.10,
			'gamma_t' : 1.50,
			'rho_t'   : 0.40,
			'mu_a'    : 0.90,
			'si_a'    : 0.10,
			'si_d'    : 0.10,
			'si_q'    : 0.10,
			'si_r'    : 0.03,
			'alpha_q' : 10.0,
			'gamma_q' : 0.1}

	# Generate parset
	pars.update(pvals)
	parname  = f"rho-{pars['rho_t']:.2f}_".replace('.', 'p')
	parname += f"alpha-{pars['alpha_t']:.1f}_".replace('.', 'p')
	parname += f"gamma-{pars['gamma_t']:.1f}_".replace('.', 'p')
	parset = {'name': parname, 'pars': pars}

	# Training oracled model
	m = goin.Model(n_hidden=n_hidden)
	m.train(parset, f'{parname}_{n_hidden}_ora', oracle=oracle, lr=lr, train_sched=(n_epochs_ora, n_batches))

	# Training non-oracled model
	m = goin.Model(n_hidden=n_hidden)
	m.train(parset, f'{parname}_{n_hidden}_mse', oracle=0.0, lr=lr, train_sched=(n_epochs_mse, n_batches))
	m.train_ctx(parset, f'{parname}_{n_hidden}_ctx', lr=0.001, train_sched=(n_epochs_ctx, n_batches))


def print_hearder(ix, pvs):
	print('\n'* 2 + 11*' ' + 48*'#' + '\n' + 11*' ' + 6*'#' + 33*' ' + 6*'#')
	print(11*' ' + 6*'#' + 3*' ' + f'computing parval comb {ix:02d}/{len(pvs):02d}' + 3*' ' + 6*'#')
	print(11*' ' + 6*'#' + 33*' ' + 6*'#' + '\n' + 11*' ' + 48*'#' + '\n')


# Parameter value combinations to test,
pvals_list = [{'rho_t': 0.75, 'alpha_t': 0.2, 'gamma_t': 5.0},
			  {'rho_t': 0.75, 'alpha_t': 0.5, 'gamma_t': 2.0},
			  {'rho_t': 0.75, 'alpha_t': 1.0, 'gamma_t': 1.0},
			  {'rho_t': 0.90, 'alpha_t': 0.2, 'gamma_t': 5.0},
			  {'rho_t': 0.90, 'alpha_t': 0.5, 'gamma_t': 2.0},
			  {'rho_t': 0.90, 'alpha_t': 1.0, 'gamma_t': 1.0},
			  {'rho_t': 0.99, 'alpha_t': 0.2, 'gamma_t': 5.0},
			  {'rho_t': 0.99, 'alpha_t': 0.5, 'gamma_t': 2.0},
			  {'rho_t': 0.99, 'alpha_t': 1.0, 'gamma_t': 1.0}]


if len(sys.argv) > 1:
	print_hearder(int(sys.argv[1]), pvs)
	run_training(pvals_list[int(sys.argv[1])])

else:
	for ix, pvals in enumerate(pvals_list):
		print_hearder(ix, pvals_list)
		run_training(pvals)

# done:


