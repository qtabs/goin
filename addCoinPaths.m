function addCoinPaths()
	coin_base       = '/home/tabs/Cloud/Projects/ContInf/Libs/repos/COIN/';
	thirdparty_base = '/home/tabs/Cloud/Projects/ContInf/Libs/thirdparty/';
	addpath(coin_base);
	addpath(thirdparty_base);
	addpath([thirdparty_base, 'npbayes-r21/utilities']);
	addpath([thirdparty_base, 'npbayes-r21/distributions/multinomial']);
	addpath([thirdparty_base, 'npbayes-r21/hdpmix']);
	addpath([thirdparty_base, 'npbayes-r21/barsdemo']);
	addpath(genpath([thirdparty_base 'lightspeed']));
	evalc('parpool()');
end