function F = fit(y, varargin)
%MONOLS.FIT Shape-constrained least-squares (or L1) fit of y against x.
%   F = monols.fit(y, 'x', x, 'order', k, 'direction', d, 'curvature', c, ...)
%   order k constrains divided differences of orders 1..k+1 (0: monotone; 1: monotone and
%   convex/concave; ...). direction: 'increasing' | 'decreasing' | 'auto' (default).
%   curvature: 'saturating' (default, e.g. decaying exponentials) | 'accelerating' | 'auto';
%   ignored for order 0. Other options: 'loss' ('l2' | 'l1'), 'weights', 'boundary' (number of
%   samples at the steep end where the highest-order difference is held at 0), 'tol', 'maxIter'.
%   A matrix y (n x p) is fit column by column and returns a struct array.
%   F has fields fitted, x, knots, coef, order, direction, curvature, loss, lossValue,
%   converged, kktResidual, nIter. Use monols.predict(F, xq) to evaluate the fit.
%   See spec/ALGORITHM.md for the exact definitions.
ip = inputParser;
ip.FunctionName = 'monols.fit';
ip.addParameter('x', []);
ip.addParameter('order', 0);
ip.addParameter('direction', 'auto');
ip.addParameter('curvature', 'saturating');
ip.addParameter('loss', 'l2');
ip.addParameter('weights', []);
ip.addParameter('boundary', 0);
ip.addParameter('tol', 1e-10);
ip.addParameter('maxIter', []);
ip.parse(varargin{:});
o = ip.Results;
validate(o);

if ~isvector(y) && ~isempty(y) %matrix: one fit per column
    for i = size(y, 2):-1:1
        args = varargin;
        if ~isempty(o.weights) && ~isvector(o.weights)
            args = [args, {'weights', o.weights(:, i)}]; %#ok<AGROW>
        end
        F(i) = monols.fit(y(:, i), args{:}); %#ok<AGROW>
    end
    return
end

n = numel(y);
P = monols.internal.prepare(y, o.x, o.weights);
if isempty(o.x), xOut = reshape(0:n-1, size(y)); else, xOut = o.x; end
if isempty(P.xu)
    F = makeFit(reshape(nan(n, 1), size(y)), xOut, zeros(0, 1), zeros(0, 1), o, o.direction, ...
        o.curvature, 0, true, 0, 0, zeros(0, 1), zeros(0, 1));
    return
end

dirs = {'increasing', 'decreasing'};
curvs = {'saturating', 'accelerating'};
if ~strcmp(o.direction, 'auto'), dirs = {o.direction}; end
if o.order == 0
    curvs = {'saturating'};
elseif ~strcmp(o.curvature, 'auto')
    curvs = {o.curvature};
end
best = [];
for d = {'increasing', 'decreasing'}
    for c = {'saturating', 'accelerating'}
        if ~any(strcmp(dirs, d{1})) || ~any(strcmp(curvs, c{1}))
            continue
        end
        if strcmp(o.loss, 'l2')
            R = monols.internal.solveShape(P, P.yu, P.wu, o.order, d{1}, c{1}, o.boundary, o.tol, o.maxIter);
        else
            R = monols.internal.irls(P, o.order, d{1}, c{1}, o.boundary, o.tol, o.maxIter);
        end
        r = R.zu(P.inverse) - P.y(P.valid);
        wv = P.w(P.valid);
        if strcmp(o.loss, 'l2'), lv = wv' * r.^2; else, lv = wv' * abs(r); end
        if isempty(best) || lv < best.lv * (1 - 1e-9)
            best = struct('lv', lv, 'd', d{1}, 'c', c{1}, 'R', R);
        end
    end
end
R = best.R;
fitted = nan(n, 1);
fitted(P.valid) = R.zu(P.inverse);
F = makeFit(reshape(fitted, size(y)), xOut, R.knots, R.coef, o, best.d, best.c, best.lv, ...
    R.converged, R.kkt, R.nIter, P.xMin + P.xSpan * P.xu, R.zu);
end

function F = makeFit(fitted, x, knots, coef, o, direction, curvature, lossValue, converged, kkt, nIter, xu, zu)
F = struct('fitted', fitted, 'x', x, 'knots', knots, 'coef', coef, 'order', o.order, ...
    'direction', direction, 'curvature', curvature, 'loss', o.loss, 'lossValue', lossValue, ...
    'converged', converged, 'kktResidual', kkt, 'nIter', nIter, 'xUnique', xu, 'fittedUnique', zu);
end

function validate(o)
isInt = @(v) isnumeric(v) && isscalar(v) && v >= 0 && v == round(v);
if ~isInt(o.order)
    error('monols:invalidOption', 'order must be a non-negative integer');
end
if ~any(strcmp(o.direction, {'increasing', 'decreasing', 'auto'}))
    error('monols:invalidOption', 'direction must be ''increasing'', ''decreasing'' or ''auto''');
end
if ~any(strcmp(o.curvature, {'saturating', 'accelerating', 'auto'}))
    error('monols:invalidOption', 'curvature must be ''saturating'', ''accelerating'' or ''auto''');
end
if ~any(strcmp(o.loss, {'l2', 'l1'}))
    error('monols:invalidOption', 'loss must be ''l2'' or ''l1''');
end
if ~isInt(o.boundary)
    error('monols:invalidOption', 'boundary must be a non-negative integer');
end
end
