function R = irls(P, order, direction, curvature, boundary, tol, maxIter)
%IRLS L1 loss by iteratively reweighted least squares (spec section 5).
yv = P.y(P.valid); wv = P.w(P.valid); inv = P.inverse; m = numel(P.xu);
R = monols.internal.solveShape(P, P.yu, P.wu, order, direction, curvature, boundary, tol, maxIter);
med = median(yv);
scale = max(abs(yv - med));
if scale == 0
    return
end
mad = median(abs(yv - med));
if mad > 0, epsilon = 1e-3 * mad; else, epsilon = 1e-3 * scale; end
r = R.zu(inv) - yv;
obj = wv' * abs(r);
best = R; bestObj = obj;
irlsConverged = false;
for it = 1:50
    [yu, wu] = monols.internal.merge(yv, wv ./ max(abs(r), epsilon), inv, m);
    R = monols.internal.solveShape(P, yu, wu, order, direction, curvature, boundary, tol, maxIter);
    r = R.zu(inv) - yv;
    newObj = wv' * abs(r);
    %relative stall test, plus an absolute floor so an exact fit (obj = 0) counts as done
    done = abs(obj - newObj) <= 1e-8 * obj + 1e-15 * scale * sum(wv);
    obj = newObj;
    if obj < bestObj
        best = R; bestObj = obj;
    end
    epsilon = max(epsilon / 2, 1e-7 * scale); %a lower floor makes IRLS stall above the optimum
    if done
        irlsConverged = true;
        break
    end
end
R = best;
R.converged = R.converged && irlsConverged;
end
