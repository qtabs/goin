function addCoinPaths()
	coin_base       = '/bcbl/home/home_a-f/clevyfidel/Workspace/COIN/';
	thirdparty_base = '/bcbl/home/home_a-f/clevyfidel/Workspace/thirdparty/';
    goin_base       = '/bcbl/home/home_a-f/clevyfidel/Workspace/goin/';
	addpath(coin_base);
	addpath(thirdparty_base);
	addpath(goin_base);
	addpath([thirdparty_base, 'npbayes-r21/utilities']);
	addpath([thirdparty_base, 'npbayes-r21/distributions/multinomial']);
	addpath([thirdparty_base, 'npbayes-r21/hdpmix']);
	addpath([thirdparty_base, 'npbayes-r21/barsdemo']);
	addpath(genpath([thirdparty_base 'lightspeed']));
	originalDir = pwd;
	cd([thirdparty_base 'lightspeed']);
	install_lightspeed;
	cd(originalDir);
	matlab.addons.toolbox.installToolbox('/bcbl/home/home_a-f/clevyfidel/Workspace/thirdparty/Truncated Multivariate Student and Normal.mltbx')
	addpath([thirdparty_base, 'MatlabProgressBar']);
	evalc('parpool()');
end
