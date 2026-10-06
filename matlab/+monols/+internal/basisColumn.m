function a = basisColumn(x, k, j)
%BASISCOLUMN Column j (1-based) of the implicit basis.
e = zeros(numel(x), 1);
e(j) = 1;
a = monols.internal.basisApply(x, k, e);
end
