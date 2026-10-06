function z = monoLS(y, normP, derN, regN, oddSign, evenSign)
%MONOLS Legacy (v1) interface, now a thin wrapper around monols.fit.
%   z = monoLS(y, normP, derN, regN, oddSign, evenSign)
%   normP:    1 or 2 (default 2) -> 'loss' 'l1' / 'l2'
%   derN:     order of the shape constraint (default 0)
%   regN:     boundary samples (default 0; ignored for derN = 0, as in v1)
%   oddSign:  >0 increasing, <0 decreasing, 0/[] automatic
%   evenSign: +1 accelerating, -1 or 0/[] saturating (v1 default)
%   y may be a matrix (fit along columns). New code should call monols.fit directly.
if nargin < 2 || isempty(normP), normP = 2; end
if nargin < 3 || isempty(derN), derN = 0; end
if nargin < 4 || isempty(regN) || derN == 0, regN = 0; end
if nargin < 5 || isempty(oddSign), oddSign = 0; end
if nargin < 6 || isempty(evenSign), evenSign = 0; end
switch normP
    case 1, loss = 'l1';
    case 2, loss = 'l2';
    otherwise
        error('monoLS:norm', 'monoLS v2 supports normP = 1 or 2 only');
end
if oddSign > 0
    direction = 'increasing';
elseif oddSign < 0
    direction = 'decreasing';
else
    direction = 'auto';
end
if evenSign > 0, curvature = 'accelerating'; else, curvature = 'saturating'; end
F = monols.fit(y, 'order', derN, 'boundary', regN, 'loss', loss, ...
    'direction', direction, 'curvature', curvature);
if numel(F) == 1
    z = F.fitted;
else
    z = [F.fitted];
end
end
