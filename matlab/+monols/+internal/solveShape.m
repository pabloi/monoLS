function R = solveShape(P, yu, wu, order, direction, curvature, boundary, tol, maxIter)
%SOLVESHAPE Weighted LS fit for one direction/curvature (spec sections 3-5).
%Returns struct with fields zu (fit per unique x), coef, knots, converged, kkt, nIter.
N = numel(P.xu);
if order == 0
    negate = strcmp(direction, 'decreasing');
    reverse = false;
else
    negate = (strcmp(direction, 'decreasing') && strcmp(curvature, 'accelerating')) || ...
             (strcmp(direction, 'increasing') && strcmp(curvature, 'saturating'));
    reverse = strcmp(curvature, 'saturating');
end
sgn = 1 - 2*negate;
if reverse
    perm = (N:-1:1)';
    xc = 1 - P.xu(perm);
else
    perm = (1:N)';
    xc = P.xu;
end
yc = sgn * yu(perm);
wc = wu(perm);
if order == 0 && boundary == 0
    zc = monols.internal.pava(yc, wc);
    coef = [zc(1); diff(zc) ./ diff(xc)];
    converged = true; kkt = 0; nIter = 0;
else
    S = monols.internal.solveCanonical(xc, yc, wc, order, boundary, tol, maxIter);
    zc = S.z; coef = S.coef; converged = S.converged; kkt = S.kkt; nIter = S.nIter;
end
zu = zeros(N, 1);
zu(perm) = sgn * zc;
ns = monols.internal.nStart(N, order);
d = coef(ns+1:end);
idx = find(d > 1e-9 * max([d; 0]));
centers = zeros(numel(idx), 1);
for i = 1:numel(idx)
    centers(i) = mean(xc(idx(i):idx(i)+order+1));
end
if reverse
    centers = 1 - centers;
end
knots = sort(P.xMin + P.xSpan * centers);
R = struct('zu', zu, 'coef', coef, 'knots', knots, 'converged', converged, 'kkt', kkt, 'nIter', nIter);
end
