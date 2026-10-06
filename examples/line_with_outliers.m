%LINE_WITH_OUTLIERS Monotone and convex fits to a noisy line with outliers, irregular x.
%   Run from the repository root after addpath('matlab').
rand('seed', 2); randn('seed', 2); %#ok<RAND>
x = sort(2*rand(80, 1) - 1);
truth = 0.7*x + 0.3;
y = truth + 0.1*randn(size(x));
idx = randperm(numel(x)); y(idx(1:5)) = randn(5, 1); %outliers
losses = {'l2', 'l1'};
for li = 1:2
    for order = 0:1
        F = monols.fit(y, 'x', x, 'order', order, 'loss', losses{li});
        fprintf('loss=%s order=%d: chose %s/%s, RMSE vs truth %.3f\n', losses{li}, order, ...
            F.direction, F.curvature, sqrt(mean((F.fitted - truth).^2)));
    end
end
F = monols.fit(y, 'x', x, 'order', 1, 'loss', 'l1');
fprintf('prediction at x = 0.5 (L1, order 1): %.3f\n', monols.predict(F, 0.5));
