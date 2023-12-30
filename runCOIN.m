
function [mu, logp, cump, lamb, a, d] = runCOIN(z, parlist, parvals)

	saveMatResults = false;

	obj = COIN;
	obj.store = {'predicted_probabilities', 'state_var', 'state_mean', 'drift', 'retention'};
	obj.runs = 1;
 	obj.max_cores = 0;
	
	if nargin == 3
		for i = 1:length(parlist)
			obj.(parlist{i}) = double(parvals{i});
		end
	end

	obj.sigma_motor_noise = sqrt(obj.sigma_sensory_noise^2 - 0.03^2);
	obj.sigma_sensory_noise = 0.03;

	% If z contains several batches:
	if size(z, 1) > 1 
		mu = zeros(size(z)); sigma = zeros(size(z));
		parfor b = 1:size(z, 1)
			[mu(b, :), logp(b, :), cump(b, :), lamb(b, :, :), a(b, :, :), d(b, :, :)] = callCoin(obj, z(b, :));
		end
	else
		[mu, logp, cump, lamb, a, d] = callCoin(obj, z);
	end

	if saveMatResults
		save('coinstate.mat', 'z', 'mu', 'logp', 'cump', 'lamb', 'a', 'd', 'parlist', 'parvals');
	end

end



function [mu, logp, cump, lambd, a, d] = callCoin(obj, z)

	obj.perturbations = squeeze(z);
	out = obj.simulate_COIN;

	mu_parts     = out.runs{1}.state_mean;
	sigma_parts  = sqrt(out.runs{1}.state_var + obj.sigma_sensory_noise^2);
	lambda_parts = out.runs{1}.predicted_probabilities;

	z_parts   = permute(repmat(z, [size(mu_parts, 1), 1, size(mu_parts, 2)]), [1, 3, 2]);
	p_z_parts = sum((lambda_parts./(sqrt(2*pi)*sigma_parts)) .* exp(-(z_parts-mu_parts).^2 ./ (2 * sigma_parts.^2)), 1);
	
	mu    = reshape(mean(sum(lambda_parts .* mu_parts, 1), 2), [1, length(z)]); 
	logp  = reshape(log(max(mean(p_z_parts, 2), eps)), [1, length(z)]);
	cump  = reshape(mean(sum(lambda_parts*0.5.*(1 + erf((z_parts - mu_parts)./(sqrt(2)*sigma_parts))), 1), 2), [1, length(z)]);
	lambd = reshape(mean(lambda_parts, 2), [size(lambda_parts, 1), length(z)]);
	a     = reshape(mean(out.runs{1}.retention, 2), [size(lambda_parts, 1), length(z)]);
	d     = reshape(mean(out.runs{1}.drift,     2), [size(lambda_parts, 1), length(z)]);

end

