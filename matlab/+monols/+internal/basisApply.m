function z = basisApply(x, k, w)
%BASISAPPLY z = A*w for the implicit basis of spec/ALGORITHM.md section 4, in O(N*k).
%Coefficient layout: [intercept; start values 1..ns-1; knots].
N = numel(x);
ns = monols.internal.nStart(N, k);
v = w(ns+1:end);
v = v(:);
for m = ns-1:-1:0
    h = x(m+2:N) - x(1:N-m-1);
    v = w(m+1) + [0; cumsum(h(:) .* v)];
end
z = v;
end
