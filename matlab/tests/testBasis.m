function testBasis()
%Basis apply/adjoint/columns (spec section 4), checked against closed forms and v1's matrix.
rand('seed', 7); randn('seed', 7); %#ok<RAND> works in MATLAB and Octave
for k = 0:4
    x = grid(25);
    A = dense(x, k);
    w = randn(25, 1); r = randn(25, 1);
    assert(max(abs(monols.internal.basisApply(x, k, w) - A*w)) < 1e-12 * max(1, max(abs(A*w))), 'apply');
    assert(max(abs(monols.internal.basisAdjoint(x, k, r) - A'*r)) < 1e-12 * max(1, max(abs(A'*r))), 'adjoint');

    x = grid(20); N = 20; K = k + 1; ns = monols.internal.nStart(N, k);
    for m = 0:ns-1
        expected = ones(N, 1);
        for l = 0:m-1, expected = expected .* (x - x(l+1)); end
        assert(max(abs(monols.internal.basisColumn(x, k, m+1) - expected)) < 1e-13, 'start column');
    end
    for j = 0:N-K-1
        expected = zeros(N, 1); i = (j+K:N-1)' + 1;
        expected(i) = x(j+K+1) - x(j+1);
        for l = j+1:j+K-1, expected(i) = expected(i) .* (x(i) - x(l+1)); end
        assert(max(abs(monols.internal.basisColumn(x, k, ns+j+1) - expected)) < 1e-13, 'knot column');
    end
end
for k = 0:3 %even grid: equal to the v1 getMatrix matrix up to positive column scaling
    n = 12; A = dense(linspace(0, 1, n)', k); L = legacyGetMatrix(n, k);
    for j = 1:n
        nz = L(:, j) ~= 0; ratio = A(nz, j) ./ L(nz, j);
        assert(all(ratio > 0) && (max(ratio) - min(ratio)) < 1e-10 * max(ratio), 'v1 column');
        assert(all(A(~nz, j) == 0), 'v1 zero pattern');
    end
end
for n = 1:3 %tiny grids
    if n == 1, x = 0; else, x = grid(n); end
    A = dense(x, 3);
    assert(isequal(size(A), [n n]) && rank(A) == n, 'tiny grid');
end
for k = 0:3 %norm proxies
    x = grid(60); v = 0.5 + 1.5*rand(60, 1);
    exact = sqrt(v' * dense(x, k).^2)';
    p = monols.internal.normProxy(x, k, v);
    ns = monols.internal.nStart(60, k);
    assert(max(abs(p(1:ns) - exact(1:ns)) ./ exact(1:ns)) < 1e-12, 'proxy start');
    ratio = p(ns+1:end) ./ exact(ns+1:end);
    assert(min(ratio) > 0.2 && max(ratio) < 5, 'proxy knots');
end
disp('testBasis: PASS')
end

function x = grid(n)
x = sort(rand(n, 1));
x = (x - x(1)) / (x(end) - x(1));
end

function A = dense(x, k)
N = numel(x); A = zeros(N);
for j = 1:N, A(:, j) = monols.internal.basisColumn(x, k, j); end
end

function A = legacyGetMatrix(n, k)
%Faithful copy of v1 incLS.m getMatrix
A = tril(ones(n));
for i = 1:k
    A(:, i+1:end) = fliplr(cumsum(fliplr(A(:, i+1:end)), 2));
end
end
