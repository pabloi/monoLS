function testPreprocess()
%Preprocessing: NaN dropping, tie merging, sorting, rescaling (spec section 2).
P = monols.internal.prepare([1; NaN; 3; 4], [], []);
assert(isequal(P.valid, [true; false; true; true]));
assertClose(P.xu, [0; 2/3; 1]);
assertClose(P.yu, [1; 3; 4]);

P = monols.internal.prepare([1; 3; 5], [2; 2; 7], [1; 3; 1]);
assertClose(P.yu, [2.5; 5]);
assertClose(P.wu, [4; 1]);
assert(isequal(P.inverse, [1; 1; 2]));

P = monols.internal.prepare([1; 2; 3], [30; 10; 20], []);
assertClose(P.xu, [0; 0.5; 1]);
assertClose(P.yu, [2; 3; 1]);
assert(P.xMin == 10 && P.xSpan == 20);

P = monols.internal.prepare(nan(4, 1), [], []);
assert(isempty(P.xu) && ~any(P.valid));

P = monols.internal.prepare([NaN; 5], [], []);
assertClose(P.xu, 0);

P = monols.internal.prepare(2*ones(5, 1), [], []);
assertClose(P.yu, 2);

bad = {[1; 0; 1], [1; -1; 1], [1; NaN; 1]};
for i = 1:numel(bad)
    try
        monols.internal.prepare([1; 2; 3], [], bad{i});
        error('testPreprocess:noError', 'invalid weights accepted');
    catch err
        assert(strcmp(err.identifier, 'monols:invalidWeights'), err.message);
    end
end

[yu, wu] = monols.internal.merge([1; 3; 5], [1; 1; 2], [1; 1; 2], 2);
assertClose(yu, [2; 5]);
assertClose(wu, [2; 2]);
disp('testPreprocess: PASS')
end

function assertClose(a, b)
assert(isequal(size(a), size(b)) || isscalar(b), 'size mismatch');
assert(all(abs(a(:) - b(:)) < 1e-12), 'values differ');
end
