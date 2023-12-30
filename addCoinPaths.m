function addCoinPaths()
	base = '/home/tabs/Cloud/Projects/ContInf/coin/';
	addpath([base, 'toolboxes/']);
	addpath([base, 'toolboxes/npbayes-r21/utilities']);
	addpath([base, 'toolboxes/npbayes-r21/distributions/multinomial']);
	addpath([base, 'toolboxes/npbayes-r21/hdpmix']);
	addpath([base, 'toolboxes/npbayes-r21/barsdemo']);
	addpath(genpath([base '/toolboxes/lightspeed']));
	addpath([base, 'toolboxes/COIN/']);
	%parpool(16);
end