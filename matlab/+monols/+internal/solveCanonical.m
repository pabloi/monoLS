function S = solveCanonical(x, y, w, k, excludedTail, tol, maxIter)
%SOLVECANONICAL Structured Lawson-Hanson NNLS over the implicit basis (spec section 5).
%Returns struct with fields z, coef, active, converged, kkt, nIter.
x = x(:); y = y(:); w = w(:);
N = numel(x);
ns = monols.internal.nStart(N, k);
if isempty(maxIter), maxIter = 10 * N; end
ybar = sum(w .* y) / sum(w);
coef = zeros(N, 1);
coef(1) = ybar;
sigma = sqrt(w' * (y - ybar).^2);
if sigma <= 1e-13 * max(abs(y)) * sqrt(sum(w)) %constant up to rounding
    S = struct('z', ybar * ones(N, 1), 'coef', coef, 'active', 1, 'converged', true, 'kkt', 0, 'nIter', 0);
    return
end

allowed = true(N, 1);
allowed(1) = false; %the intercept is always active
if excludedTail > 0
    allowed(max(ns, N - excludedTail)+1:end) = false;
end
p = monols.internal.normProxy(x, k, w);
allowed = allowed & p > 0;
p(p <= 0) = 1;
sw = sqrt(w);
cols = zeros(N, 0); colNorms = zeros(1, 0); colIdx = zeros(1, 0); %cache of built columns

P = 1;
z = ybar * ones(N, 1);
banned = false(N, 1);
nIter = 0;
while nIter < maxIter
    nIter = nIter + 1;
    score = monols.internal.basisAdjoint(x, k, w .* (y - z)) ./ p;
    cand = allowed;
    cand(P) = false;
    cand(banned) = false;
    if ~any(cand), break; end
    sc = score;
    sc(~cand) = -Inf;
    [best, j] = max(sc);
    if best <= tol * sigma, break; end
    P(end+1) = j; %#ok<AGROW>
    while true
        u = leastSquares(P);
        cur = coef(P);
        bad = find(u(2:end) <= 0) + 1;
        if isempty(bad)
            coef(P) = u;
            break
        end
        [alpha, ib] = min(cur(bad) ./ (cur(bad) - u(bad)));
        coef(P) = cur + alpha * (u - cur);
        coef(P(bad(ib))) = 0; %exact zero: rounding can leave ~1e-16 and cycle forever
        %compare contributions coef*||column||: coefficients themselves span many decades
        [~, loc] = ismember(P(2:end), colIdx);
        contrib = coef(P(2:end)) .* colNorms(loc)';
        floorValue = 1e-14 * max([abs(contrib); 0]);
        keep = [true; contrib > floorValue];
        coef(P(~keep)) = 0;
        P = P(keep);
    end
    if any(P == j)
        banned(:) = false;
    else
        banned(j) = true; %entered and left immediately: numerically degenerate, skip for now
    end
    z = monols.internal.basisApply(x, k, coef);
end

score = monols.internal.basisAdjoint(x, k, w .* (y - z)) ./ p;
outside = allowed;
outside(P) = false;
inside = false(N, 1);
inside(P(2:end)) = true;
kkt = max([0; max(score(outside), 0); abs(score(inside))]) / sigma;
S = struct('z', z, 'coef', coef, 'active', sort(P(:)), 'converged', kkt <= tol, 'kkt', kkt, 'nIter', nIter);

    function u = leastSquares(P)
        for q = P(:)'
            if ~any(colIdx == q)
                c = monols.internal.basisColumn(x, k, q);
                cols(:, end+1) = c; %#ok<AGROW>
                colNorms(end+1) = norm(sw .* c); %#ok<AGROW>
                colIdx(end+1) = q; %#ok<AGROW>
            end
        end
        [~, loc] = ismember(P, colIdx);
        B = (sw .* cols(:, loc)) ./ colNorms(loc);
        u = (B \ (sw .* y)) ./ colNorms(loc)';
    end
end
