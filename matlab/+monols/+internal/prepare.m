function P = prepare(y, x, w)
%PREPARE Drop NaN, sort, merge tied x, rescale x to [0,1] (spec/ALGORITHM.md section 2).
y = double(y(:));
n = numel(y);
if isempty(x), x = (0:n-1)'; else, x = double(x(:)); end
if isempty(w), w = ones(n, 1); else, w = double(w(:)); end
if numel(x) ~= n || numel(w) ~= n
    error('monols:invalidInput', 'x and weights must have the same length as y');
end
if any(~isfinite(w)) || any(w <= 0)
    error('monols:invalidWeights', 'weights must be finite and > 0');
end
valid = ~(isnan(x) | isnan(y));
[xs, ~, inverse] = unique(x(valid));
P = struct('xu', zeros(0, 1), 'yu', zeros(0, 1), 'wu', zeros(0, 1), 'inverse', inverse(:), ...
           'valid', valid, 'xMin', 0, 'xSpan', 1, 'y', y, 'w', w);
if isempty(xs)
    return
end
[P.yu, P.wu] = monols.internal.merge(y(valid), w(valid), P.inverse, numel(xs));
P.xMin = xs(1);
if numel(xs) > 1
    P.xSpan = xs(end) - xs(1);
end
P.xu = (xs(:) - P.xMin) / P.xSpan;
end
