function g = basisAdjoint(x, k, r)
%BASISADJOINT g = A'*r for the implicit basis, in O(N*k).
N = numel(x);
ns = monols.internal.nStart(N, k);
g = zeros(N, 1);
u = r(:);
for m = 0:ns-1
    g(m+1) = sum(u);
    t = flipud(cumsum(flipud(u)));
    h = x(m+2:N) - x(1:N-m-1);
    u = h(:) .* t(2:end);
end
g(ns+1:end) = u;
end
