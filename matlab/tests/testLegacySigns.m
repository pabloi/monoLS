function testLegacySigns()
%Matrix input must honor the requested signs exactly like vector input.
i = (0:29)';
y = 10 - 0.3*i + 0.2*sin(i); %clearly decreasing data
zVec = monoLS(y, 2, 1, 0, 1, -1); %force an increasing, concave fit
zMat = monoLS([y y], 2, 1, 0, 1, -1);
assert(max(abs(zMat(:,1) - zVec)) < 1e-10, 'matrix column 1 differs from vector fit');
assert(max(abs(zMat(:,2) - zVec)) < 1e-10, 'matrix column 2 differs from vector fit');
assert(all(diff(zVec) >= -1e-10), 'requested increasing fit is not increasing');
disp('testLegacySigns: PASS')
end
