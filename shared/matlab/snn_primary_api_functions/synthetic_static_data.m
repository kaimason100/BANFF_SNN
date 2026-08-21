% synthetic_static_data.m
function data = synthetic_static_data(domain, opts)
N = 8;
Din = 3;
X = single(linspace(-1,1,N*Din));
X = reshape(X, N, Din);
switch domain
    case "classification"
        labels = 1 + mod((1:N)', 2);
        Y = zeros(N,2,'single');
        for i = 1:N, Y(i,labels(i)) = 1; end
    case "regression"
        Y = single(0.25*X(:,1) - 0.5*X(:,2) + 0.1*X(:,3));
    otherwise
        error('Unknown synthetic domain.');
end
opts.seed = get_opt(opts, 'seed', 42);
data = split_and_standardize(X, Y, opts);
end

