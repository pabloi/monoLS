function yq = predict(F, xq)
%MONOLS.PREDICT Evaluate a fit from monols.fit at new x values.
%   Linear interpolation between fitted samples. Outside the data range: constant for
%   order 0, linear extension of the end segment for order >= 1.
xu = F.xUnique;
zu = F.fittedUnique;
yq = nan(size(xq));
if isempty(xu)
    return
end
if numel(xu) == 1
    yq(:) = zu;
    return
end
yq = interp1(xu, zu, xq, 'linear');
lo = xq < xu(1);
hi = xq > xu(end);
if F.order == 0
    yq(lo) = zu(1);
    yq(hi) = zu(end);
else
    yq(lo) = zu(1) + (xq(lo) - xu(1)) * (zu(2) - zu(1)) / (xu(2) - xu(1));
    yq(hi) = zu(end) + (xq(hi) - xu(end)) * (zu(end) - zu(end-1)) / (xu(end) - xu(end-1));
end
end
