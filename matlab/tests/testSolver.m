function testSolver()
%Structured active-set solver vs a dense column-scaled NNLS reference (spec section 5).
for k = 1:4
    [x, y, w] = data(200, k, true);
    S = monols.internal.solveCanonical(x, y, w, k, 0, 1e-10, []);
    assert(max(abs(S.z - reference(x, y, w, k))) < 1e-8 * (max(y) - min(y)), sprintf('k=%d mismatch', k));
    assert(S.converged && S.kkt <= 1e-10, 'not converged');
end
for k = 0:2
    [x, y, w] = data(80, 10 + k, false);
    S = monols.internal.solveCanonical(x, y, w, k, 0, 1e-10, []);
    assert(max(abs(S.z - reference(x, y, w, k))) < 1e-8 * (max(y) - min(y)), 'even grid mismatch');
end
S = monols.internal.solveCanonical(linspace(0, 1, 10)', 3*ones(10, 1), ones(10, 1), 2, 0, 1e-10, []);
assert(max(abs(S.z - 3)) < 1e-14 && S.converged, 'constant data');
x = linspace(0, 1, 30)'; y = 1 + x + x.^3;
S = monols.internal.solveCanonical(x, y, ones(30, 1), 2, 0, 1e-10, []);
assert(max(abs(S.z - y)) < 1e-10, 'in-cone data not reproduced');
%Regression: blocking coefficient left at ~1e-16 used to cycle forever (fixture even_order3)
cases = jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'fixtures', 'cases.json')));
c = cases(strcmp({cases.name}, 'even_order3'));
y = -flipud(c.y(:)); n = numel(y);
x = 1 - flipud((0:n-1)' / (n - 1));
S = monols.internal.solveCanonical(x, y, ones(n, 1), 3, 0, 1e-10, []);
assert(S.converged, 'cycling case did not converge');
%Large n without a dense matrix
[x, y, w] = data(20000, 99, false);
t = tic;
S = monols.internal.solveCanonical(x, y, w, 2, 0, 1e-10, []);
assert(toc(t) < 30 && S.converged, 'large n too slow or not converged');
%Regression: a drop floor relative to max(coef) zeroed valid coefficients of tiny columns
%(coefficients spanned 1e-1..1e15 at n = 1e5, order 3) and the solver cycled
n = 100000; randn('seed', 1); %#ok<RAND>
x = linspace(0, 1, n)'; y = 1 - exp(-5*x) + 0.1*randn(n, 1);
S = monols.internal.solveCanonical(1 - flipud(x), -flipud(y), ones(n, 1), 3, 0, 1e-10, 300);
assert(S.converged, 'n = 1e5, order 3 did not converge');
disp('testSolver: PASS')
end

function [x, y, w] = data(n, seed, irregular)
rand('seed', seed); randn('seed', seed); %#ok<RAND>
if irregular, x = sort(rand(n, 1)); else, x = linspace(0, 1, n)'; end
x = (x - x(1)) / (x(end) - x(1));
y = exp(3*x) + 0.5*randn(n, 1);
w = 0.3 + 2.7*rand(n, 1);
end

function z = reference(x, y, w, k)
N = numel(x); A = zeros(N);
for j = 1:N, A(:, j) = monols.internal.basisColumn(x, k, j); end
A = [A(:, 1), -A(:, 1), A(:, 2:end)];
B = sqrt(w) .* A;
s = sqrt(sum(B.^2, 1));
c = lsqnonneg(B ./ s, sqrt(w) .* y);
z = A * (c ./ s');
end
