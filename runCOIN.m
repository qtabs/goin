
function [mu, logp, cump, lamb, a, d] = runCOIN(y, parlist, parvals)

	for b = 1:size(y, 1)
		[mu(b, :), logp(b, :), cump(b, :), lamb(b, :, :), a(b, :, :), d(b, :, :)] = call_coin(y(b, :), parlist, parvals);
	end

end


function [mu, logp, cump, lamb, a, d] = call_coin(y, parlist, parvals, nruns)

	if nargin < 4
		nruns = 10;
	end

	if nargin == 1
		parlist = {};
		parvals = {};
	end

	y   = squeeze(y);
	obj = instantiate_coin(parlist, parvals);

	obj.runs = nruns;
	obj.max_cores = nruns;
	obj.perturbations = y;
	
	out  = obj.simulate_COIN;
	mu   = zeros(length(y), nruns);
	cump = zeros(length(y) * nruns, 1);
	logp = zeros(length(y), nruns);
	lamb = zeros(obj.max_contexts + 1, length(y), nruns);
	a    = zeros(obj.max_contexts + 1, length(y), nruns);
	d    = zeros(obj.max_contexts + 1, length(y), nruns);

	for i = 1:nruns

		mu_parts     = out.runs{i}.state_mean;
		sigma_parts  = sqrt(out.runs{i}.state_var + obj.sigma_sensory_noise^2);
		lambda_parts = out.runs{i}.predicted_probabilities;

		y_parts   = permute(repmat(y, [size(mu_parts, 1), 1, size(mu_parts, 2)]), [1, 3, 2]);
		p_y_parts = sum((lambda_parts./(sqrt(2*pi)*sigma_parts)) .* exp(-(y_parts-mu_parts).^2 ./ (2 * sigma_parts.^2)), 1);
		
		mu(:, i)      = reshape(mean(sum(lambda_parts .* mu_parts, 1), 2), [1, length(y)]); 
		logp(:, i)    = reshape(log(max(mean(p_y_parts, 2), eps)), [1, length(y)]);
		lamb(:, :, i) = reshape(mean(lambda_parts, 2), [size(lambda_parts, 1), length(y)]);
		a(:, :, i)    = reshape(mean(out.runs{i}.retention, 2), [size(lambda_parts, 1), length(y)]);
		d(:, :, i)    = reshape(mean(out.runs{i}.drift,     2), [size(lambda_parts, 1), length(y)]);
		cump = [cump; reshape(mean(sum(lambda_parts*0.5.*(1 + erf((y_parts - mu_parts)./(sqrt(2)*sigma_parts))), 1), 2), [length(y), 1])];

	end

	mu   = mean(mu, 2);
	logp = max(logp(:)) + log(sum(exp(logp - max(logp(:))), 2)) - log(nruns);
	lamb = mean(lamb, 3);
	a = mean(a, 3);
	d = mean(d, 3);

end


function obj = instantiate_coin(parlist, parvals)

	if nargin == 0
		parlist = {};
		parvals = {};
	end

	obj = COIN;

	for i = 1:length(parlist)
		obj.(parlist{i}) = double(parvals{i});
	end

	obj.store = {'predicted_probabilities', 'state_var', 'state_mean', 'drift', 'retention', 'average_state'};
	obj.runs = 1;
 	obj.max_cores = 0;
 	obj.add_observation_noise = false;
	obj.verbose = false;
	obj.particles = 100;
	obj.sigma_motor_noise = sqrt(obj.sigma_sensory_noise^2 - 0.03^2);
	obj.sigma_sensory_noise = 0.03;

end
