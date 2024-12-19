function addCoinPaths()
	coin_base       = 'C:/Users/cleme/OneDrive/Documents/BCBL/Workspace/COIN/';
	thirdparty_base = 'C:/Users/cleme/OneDrive/Documents/BCBL/Workspace/thirdparty/';
    goin_base       = 'C:/Users/cleme/OneDrive/Documents/BCBL/Workspace/goin/';
	addpath(coin_base);
	addpath(thirdparty_base);
	addpath(goin_base);
	addpath([thirdparty_base, 'npbayes-r21/utilities']);
	addpath([thirdparty_base, 'npbayes-r21/distributions/multinomial']);
	addpath([thirdparty_base, 'npbayes-r21/hdpmix']);
	addpath([thirdparty_base, 'npbayes-r21/barsdemo']);
	addpath(genpath([thirdparty_base 'lightspeed']));
	evalc('parpool()');
end
