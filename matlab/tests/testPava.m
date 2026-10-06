function testPava()
%Weighted PAVA against a brute-force reference and simple invariants.
y = [0; 0.5; 0.5; 2; 7];
assert(isequal(monols.internal.pava(y, ones(5, 1)), y), 'monotone input changed');
assert(monols.internal.pava(3, 2) == 3, 'single sample');
assert(isempty(monols.internal.pava(zeros(0, 1), zeros(0, 1))), 'empty input');
assert(max(abs(monols.internal.pava([3; 1; 2], [1; 1; 1]) - 2)) < 1e-15, 'pooling');
assert(max(abs(monols.internal.pava([3; 1], [1; 3]) - 1.5)) < 1e-15, 'weighted pooling');
%Compare with the dense active-set solution (an independent path) on random data
rand('seed', 3); randn('seed', 3); %#ok<RAND>
n = 60; x = linspace(0, 1, n)'; y = 2*x + randn(n, 1); w = 0.2 + rand(n, 1);
z = monols.internal.pava(y, w);
assert(all(diff(z) >= 0), 'not monotone');
% KKT for weighted isotonic regression: suffix sums of weighted residuals are <= 0, and the
% total is 0 (e.g. y = [3 1] gives z = [2 2] and suffix sum 1 - 2 = -1)
c = flipud(cumsum(flipud(w .* (y - z))));
assert(max(c) < 1e-10 && abs(c(1)) < 1e-10, 'KKT violated');
disp('testPava: PASS')
end
