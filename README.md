This is a provisional repository for the GRU-COIN project. All code is provisional and not guarateed.
Running the COIN inference algorithm requires installing the COIN model, available [here](https://github.com/jamesheald/COIN). Check addCoinPaths.m and correct the paths so that they point towards the correct all dependencies

Contents
------------------------
coin.py : Generative model for the [COIN model](https://www.nature.com/articles/s41586-021-04129-3)
	Provides for three main classes:
		GenerativeModel() : Base class 
		ExplicitGenerativeModel(GenerativeModel) : Generative model that generates data by first sampling the transition probability matrices for context and cue transitions using GEM and DP. This construction is an approximation, since the transisition probability matrices should have infinite dimension.
		CRFGenerativeModel(GenerativeModel) : Generative model that generates data using the Chinese Restaurant Franchise constructions. This construction is exact.
		Cue emissions should work in principle in both versions but they have not been tested.
	In addition it provides for secondary functions that can be used to benchmark and interact with the COIN inference model. 

goin.py : Set of classes and functions that fit GRU-RNNs to predict data sampled from the COIN generative model.
	Provides for two classes:
		GruRNN(torch.nn.Module): Pytorch-specified recurrent neural network with the input and output linear layers necessary to perform inference over samples of the COIN Generative Model.
		Model(): General and extensible class to fit pytorch modules to the COIN Generative Model. Assumes that the output of the pytorch module encodes the mean and standard deviation of a Gaussian distribution.
	In addition it provides for secondary functions that can be used to perform group fits and compare the performance with the COIN inference model.

noin.py: a clone of goin.py handling mixture of Gaussian predictive distributions.

example.py: a script displaying an example training pipeline

addCoinPaths.m: MATLAB script called from coin.py that sets up all the paths required to run the COIN model

runCOIN.m: MATLAB function that acts as interface between coin.py and the COIN inference algorithm.

subParameters.csv: CSV textfile listing all the hyperparametrisations published in the original COIN paper.

expsgroupdata.pickle: pickle containing the experimental data for the experiments used to evaluate the COIN inference model in the original paper that do not include cue emissions.
