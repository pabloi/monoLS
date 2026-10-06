function [yu, wu] = merge(yValid, wValid, inverse, m)
%MERGE Weighted mean of y and summed weight per group (groups given by inverse).
wu = accumarray(inverse(:), wValid(:), [m 1]);
yu = accumarray(inverse(:), wValid(:) .* yValid(:), [m 1]) ./ wu;
end
