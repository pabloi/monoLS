function p = normProxy(x, k, v)
%NORMPROXY Weighted column norms: exact for start columns, an estimate for knot columns.
%A knot column is a degree-k polynomial on its support growing to |A(N,j)|, so its weighted
%norm is about |A(N,j)| * sqrt(sum(v over support) / (2k+1)).
N = numel(x);
ns = monols.internal.nStart(N, k);
p = zeros(N, 1);
for m = 1:ns
    p(m) = sqrt(v(:)' * monols.internal.basisColumn(x, k, m).^2);
end
if N > ns
    e = zeros(N, 1);
    e(end) = 1;
    lastRow = monols.internal.basisAdjoint(x, k, e);
    supportW = flipud(cumsum(flipud(v(:))));
    p(ns+1:end) = abs(lastRow(ns+1:end)) .* sqrt(supportW(k+2:end) / (2*k + 1));
end
end
