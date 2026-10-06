function testFixtures()
%Golden-fixture parity with the independent reference (tests/fixtures/cases.json).
here = fileparts(mfilename('fullpath'));
cases = jsondecode(fileread(fullfile(here, '..', '..', 'tests', 'fixtures', 'cases.json')));
if iscell(cases), cases = [cases{:}]; end
for i = 1:numel(cases)
    c = cases(i); o = c.options;
    y = vec(c.y);
    F = monols.fit(y, 'x', vec(c.x), 'weights', vec(c.weights), 'order', o.order, ...
        'direction', o.direction, 'curvature', o.curvature, 'loss', o.loss, 'boundary', o.boundary);
    z = vec(c.expected.fitted);
    yv = y(~isnan(y));
    span = 1;
    if ~isempty(yv) && max(yv) > min(yv), span = max(yv) - min(yv); end
    assert(isequal(isnan(F.fitted(:)), isnan(z)), [c.name ': NaN pattern']);
    if strcmp(o.loss, 'l2') && any(~isnan(z)) %L1 minimizers need not be unique: compare the loss only
        assert(max(abs(F.fitted(~isnan(z)) - z(~isnan(z)))) <= 1e-8 * span, [c.name ': fitted']);
    end
    assert(strcmp(F.direction, c.expected.direction) && strcmp(F.curvature, c.expected.curvature), ...
        [c.name ': chosen shape']);
    rel = 1e-7;
    if strcmp(o.loss, 'l1'), rel = 1e-6; end
    assert(abs(F.lossValue - c.expected.loss_value) <= rel * abs(c.expected.loss_value) + 1e-12, ...
        [c.name ': loss']);
end
disp('testFixtures: PASS')
end

function v = vec(a)
if iscell(a) %all-null arrays may decode to a cell of empties in MATLAB
    v = nan(numel(a), 1);
    for i = 1:numel(a), if ~isempty(a{i}), v(i) = a{i}; end, end
else
    v = double(a(:));
end
end
