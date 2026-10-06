function testLegacyNaN()
%Automatic direction detection must ignore NaN samples.
i = (0:29)';
y = 1 + 0.5*i + 0.3*cos(3*i); %increasing data
y([4 11 20]) = NaN;
z = monoLS(y, 2, 0); %direction auto-detected
assert(isequal(isnan(z), isnan(y)), 'NaN positions not preserved');
zz = z(~isnan(z));
assert(all(diff(zz) >= -1e-10), 'increasing data got a non-increasing fit');
assert(zz(end) > zz(1), 'fit is not increasing overall');
disp('testLegacyNaN: PASS')
end
