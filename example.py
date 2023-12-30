import goin
# Importing coin separately introduces a problem loading torch models (not sure why)


# Load the set of subject-specific hyperparameters:
parsets = goin.coin.load_sub_pars(types = ['S', 'E'])
oracle  = 0.5
dweight = 0.2


# Train a baseline model on one of the hyperparameter sets with rich transition matrices
m = goin.Model()
baseline = f'pars-transitions_oracle-{100*w:.0f}'
m.train('transitions', baseline, oracle=oracle, lr=0.01)


# Train a baseline-initialised model independently for each subject
for sub in parsets.keys():
	print(f'Training for {sub}')

	# Train on cross entropy on samples drawn from the coin GM
	m = goin.Model()
	m.load_weights(goin.find_latest(baseline))
	modname = f'pars-{sub}_pretrained-transitions_oracle-{100*w:.0f}_trained'
	m.train(sub, modname, oracle=oracle, lr=0.01)
	
	# Further train on datafit
	modname = f'pars-{sub}_pretrained-transitions_oracle-{100*w:.0f}_trained_tuned'
	m.tune(sub, modname, oracle=oracle, dweight=dweight, lr=0.005)


# Check the fit of the average prediction of the models across subjects to the experimental data
modkey = f'pretrained-transitions_oracle-{100*w:.0f}_trained_tuned'
goin.summary_stats_fit(modkey)