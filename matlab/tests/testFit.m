function testFit()
%Properties of the public fit/predict API (mirrors python/tests/test_fit.py).
randn('seed', 3); rand('seed', 3); %#ok<RAND>
curvs = {'saturating', 'accelerating'};
y = 2 + 0.5*(0:29)';
for order = 0:4
    for c = 1:2
        F = monols.fit(y, 'order', order, 'direction', 'increasing', 'curvature', curvs{c});
        assert(max(abs(F.fitted - y)) < 1e-9, 'line not reproduced');
    end
end
y = log1p((0:39)') + 0.1*randn(40, 1);
for order = 0:2
    up = monols.fit(y, 'order', order, 'direction', 'increasing', 'curvature', 'saturating');
    down = monols.fit(-y, 'order', order, 'direction', 'decreasing', 'curvature', 'saturating');
    assert(max(abs(down.fitted + up.fitted)) < 1e-9, 'mirror symmetry');
    x = sort(5*rand(30, 1)); yy = sqrt(x) + 0.1*randn(30, 1);
    a = monols.fit(yy, 'x', x, 'order', order); b = monols.fit(yy, 'x', 3*x + 7, 'order', order);
    assert(max(abs(a.fitted - b.fitted)) < 1e-9, 'affine invariance');
end
x = sort(rand(50, 1));
F = monols.fit(exp(-4*x) + 0.05*randn(50, 1), 'x', x, 'order', 2, 'direction', 'decreasing', 'curvature', 'saturating');
d1 = diff(F.fitted) ./ diff(x); d2 = diff(d1) ./ (x(3:end) - x(1:end-2));
assert(max(d1) <= 1e-9 && min(d2) >= -1e-6, 'fit outside cone');
F = monols.fit(4.2*ones(12, 1), 'order', 2);
assert(max(abs(F.fitted - 4.2)) < 1e-14 && F.converged, 'constant');
yt = [1; 3; 2];
for n = 1:3
    F = monols.fit(yt(1:n), 'order', 3);
    assert(all(isfinite(F.fitted)) && F.converged, 'tiny input');
end
F = monols.fit(nan(5, 1), 'order', 1);
assert(all(isnan(F.fitted)) && F.lossValue == 0 && strcmp(F.direction, 'auto'), 'all NaN');
assert(all(isnan(monols.predict(F, [0; 1]))), 'all NaN predict');
yr = (1:6) + 0.1*randn(1, 6); %row vector is one series, not six
F = monols.fit(yr, 'order', 0);
assert(isequal(size(F.fitted), size(yr)) && numel(F) == 1, 'row vector');
t = tic;
F = monols.fit(exp(linspace(0, 3, 20000)') + 0.5*randn(20000, 1), 'order', 2, ...
    'direction', 'increasing', 'curvature', 'accelerating');
assert(toc(t) < 30 && F.converged, 'large n');
x = (0:10)';
F = monols.fit(max(0, x - 5), 'x', x, 'order', 1, 'direction', 'increasing', 'curvature', 'accelerating');
assert(isequal(size(F.knots), [1 1]) && abs(F.knots - 5) < 1e-12, 'knots');
x = (0:3)';
F0 = monols.fit(x, 'x', x, 'order', 0, 'direction', 'increasing');
assert(max(abs(monols.predict(F0, [-1; 1.5; 9]) - [0; 1.5; 3])) < 1e-12, 'predict order 0');
F1 = monols.fit(x, 'x', x, 'order', 1, 'direction', 'increasing');
assert(max(abs(monols.predict(F1, [-1; 1.5; 5]) - [-1; 1.5; 5])) < 1e-12, 'predict order 1');
F = monols.fit([NaN; 2], 'order', 1);
assert(isequal(monols.predict(F, [0; 10]), [2; 2]), 'predict single sample');
Y = [(0:9)', -(0:9)'];
F = monols.fit(Y, 'order', 0);
G = monols.fit(Y(:, 2), 'order', 0);
assert(numel(F) == 2 && max(abs(F(2).fitted - G.fitted)) < 1e-14, 'matrix input');
bad = {{'order', -1}, {'direction', 'up'}, {'curvature', 'flat'}, {'loss', 'l3'}, {'boundary', -2}};
for i = 1:numel(bad)
    try
        monols.fit((1:5)', bad{i}{:});
        error('testFit:noError', 'invalid option accepted');
    catch err
        assert(strcmp(err.identifier, 'monols:invalidOption'), err.message);
    end
end
for ob = [0 3; 1 2; 2 4]' %boundary zeroes the last highest-order divided differences
    order = ob(1); b = ob(2);
    x = linspace(0, 1, 40)'; randn('seed', order); %#ok<RAND>
    yb = exp(3*x) + 0.3*randn(40, 1); yb(end) = yb(end) + 3;
    Fb = monols.fit(yb, 'x', x, 'order', order, 'direction', 'increasing', 'curvature', 'accelerating', 'boundary', b);
    Ff = monols.fit(yb, 'x', x, 'order', order, 'direction', 'increasing', 'curvature', 'accelerating');
    z = Fb.fitted; free = Ff.fitted;
    d = z;
    for q = 1:order+1, d = (d(2:end) - d(1:end-1)) ./ (x(q+1:end) - x(1:end-q)); end
    assert(max(abs(d(end-b+1:end))) <= 1e-8 * max(abs(d)), 'boundary differences');
    assert(max(abs(z - free)) > 1e-6, 'boundary had no effect');
end
disp('testFit: PASS')
end
