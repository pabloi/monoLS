function bench(maxN)
%BENCH Timing of monols.fit for growing n and order (prints a markdown table).
%   Run from the repository root: addpath('matlab'); addpath('benchmarks'); bench
if nargin < 1, maxN = 1e5; end
randn('seed', 0); %#ok<RAND>
fprintf('| n | order 0 | order 1 | order 2 | order 3 |\n|---|---|---|---|---|\n');
for n = [1e3 1e4 1e5]
    if n > maxN, break; end
    x = linspace(0, 1, n)';
    y = 1 - exp(-5*x) + 0.1*randn(n, 1);
    cells = cell(1, 4);
    for k = 0:3
        t = tic;
        F = monols.fit(y, 'x', x, 'order', k, 'direction', 'increasing', 'curvature', 'saturating');
        flag = '';
        if ~F.converged, flag = ' (!)'; end
        cells{k+1} = sprintf('%.3f s%s', toc(t), flag);
    end
    fprintf('| %d | %s | %s | %s | %s |\n', n, cells{:});
end
%Cost grows with the number of knots: nearly noise-free data needs many of them.
fprintf('\nHigh SNR (noise sd 1e-3): time [knots]\n| n | order 1 | order 2 | order 3 |\n|---|---|---|---|\n');
for n = [1e4 3e4]
    if n > maxN, break; end
    x = linspace(0, 1, n)';
    y = 1 - exp(-5*x) + 1e-3*randn(n, 1);
    cells = cell(1, 3);
    for k = 1:3
        t = tic;
        F = monols.fit(y, 'x', x, 'order', k, 'direction', 'increasing', 'curvature', 'saturating');
        cells{k} = sprintf('%.2f s [%d]', toc(t), numel(F.knots));
    end
    fprintf('| %d | %s | %s | %s |\n', n, cells{:});
end
end
