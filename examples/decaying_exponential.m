%DECAYING_EXPONENTIAL Fit a noisy two-timescale decay with increasingly strict shapes.
%   Run from the repository root after addpath('matlab').
randn('seed', 1); %#ok<RAND>
t = (0:59)';
truth = 1.5*exp(-t/4) + 0.8*exp(-t/25) + 0.2;
y = truth + 0.08*randn(size(t));
y([8 32]) = y([8 32]) + 0.8; %two outliers

names = {'monotone (order 0)', '+ convex (order 1)', '+ convex, L1 loss', 'order 2, boundary=2'};
F = [monols.fit(y, 'x', t, 'order', 0, 'direction', 'decreasing'), ...
     monols.fit(y, 'x', t, 'order', 1, 'direction', 'decreasing'), ...
     monols.fit(y, 'x', t, 'order', 1, 'direction', 'decreasing', 'loss', 'l1'), ...
     monols.fit(y, 'x', t, 'order', 2, 'direction', 'decreasing', 'boundary', 2)];
for i = 1:numel(F)
    fprintf('%-22s error vs truth (RMSE) = %.4f   %d knots\n', names{i}, ...
        sqrt(mean((F(i).fitted - truth).^2)), numel(F(i).knots));
end

figure; hold on
plot(t, y, '.', 'Color', [.5 .5 .5], 'DisplayName', 'data');
plot(t, truth, 'k--', 'DisplayName', 'truth');
for i = 1:numel(F)
    plot(t, F(i).fitted, 'DisplayName', names{i});
end
legend show; title('Shape-constrained fits of a decaying curve');
