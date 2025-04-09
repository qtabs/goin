function addCoinPaths()
    coin_base       = '/bcbl/home/home_a-f/clevyfidel/Workspace/COIN/';
    thirdparty_base = '/bcbl/home/home_a-f/clevyfidel/Workspace/thirdparty/';
    goin_base       = '/bcbl/home/home_a-f/clevyfidel/Workspace/goin/';
    
    % Add paths if they are not already added
    if ~contains(path, coin_base)
        addpath(coin_base);
    end
    
    if ~contains(path, thirdparty_base)
        addpath(thirdparty_base);
    end
    
    if ~contains(path, goin_base)
        addpath(goin_base);
    end
    
    if ~contains(path, [thirdparty_base, 'npbayes-r21/utilities'])
        addpath([thirdparty_base, 'npbayes-r21/utilities']);
    end
    
    if ~contains(path, [thirdparty_base, 'npbayes-r21/distributions/multinomial'])
        addpath([thirdparty_base, 'npbayes-r21/distributions/multinomial']);
    end
    
    if ~contains(path, [thirdparty_base, 'npbayes-r21/hdpmix'])
        addpath([thirdparty_base, 'npbayes-r21/hdpmix']);
    end
    
    if ~contains(path, [thirdparty_base, 'npbayes-r21/barsdemo'])
        addpath([thirdparty_base, 'npbayes-r21/barsdemo']);
    end
    
    if ~contains(path, [thirdparty_base 'lightspeed'])
        addpath(genpath([thirdparty_base 'lightspeed']));
        % Check if lightspeed is already installed
        originalDir = pwd;
        cd([thirdparty_base 'lightspeed']);
        if exist('install_lightspeed.m', 'file') == 2
            install_lightspeed;
        end
        cd(originalDir);
    end
    
    % Install toolbox only if not already installed
    toolboxPath = '/bcbl/home/home_a-f/clevyfidel/Workspace/thirdparty/Truncated Multivariate Student and Normal.mltbx';
    if ~exist(toolboxPath, 'file')
        matlab.addons.toolbox.installToolbox(toolboxPath);
    end
    
    % Add MatlabProgressBar path if not already added
    if ~contains(path, [thirdparty_base, 'MatlabProgressBar'])
        addpath([thirdparty_base, 'MatlabProgressBar']);
    end
    
    % Start parallel pool only if it's not already running
    try
        evalc('parpool()');
    catch
        % Handle the case where parallel pool cannot be started
        warning('Unable to start parallel pool.');
    end
end



% function addCoinPaths()
% 	coin_base       = '/bcbl/home/home_a-f/clevyfidel/Workspace/COIN/';
% 	thirdparty_base = '/bcbl/home/home_a-f/clevyfidel/Workspace/thirdparty/';
%     goin_base       = '/bcbl/home/home_a-f/clevyfidel/Workspace/goin/';
% 	addpath(coin_base);
% 	addpath(thirdparty_base);
% 	addpath(goin_base);
% 	addpath([thirdparty_base, 'npbayes-r21/utilities']);
% 	addpath([thirdparty_base, 'npbayes-r21/distributions/multinomial']);
% 	addpath([thirdparty_base, 'npbayes-r21/hdpmix']);
% 	addpath([thirdparty_base, 'npbayes-r21/barsdemo']);
% 	addpath(genpath([thirdparty_base 'lightspeed']));
% 	originalDir = pwd;
% 	cd([thirdparty_base 'lightspeed']);
% 	install_lightspeed;
% 	cd(originalDir);
% 	matlab.addons.toolbox.installToolbox('/bcbl/home/home_a-f/clevyfidel/Workspace/thirdparty/Truncated Multivariate Student and Normal.mltbx');
% 	addpath([thirdparty_base, 'MatlabProgressBar']);
% 	evalc('parpool()');
% end
