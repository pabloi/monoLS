function testLegacyLine()
%Regression test for the legacy monoLS: data that already satisfies the
%shape constraints must be returned unchanged, for every order.
i = (0:39)';
line = 2 + 0.5*i;
curve = 1 + i + 0.01*i.^3/6; %increasing, convex, with increasing curvature
for derN = 0:3
    z = monoLS(line, 2, derN, 0, 1, 1);
    assert(max(abs(z - line)) < 1e-6, sprintf('line not reproduced at order %d', derN));
end
for derN = 0:2
    z = monoLS(curve, 2, derN, 0, 1, 1);
    assert(max(abs(z - curve)) < 1e-6, sprintf('in-cone curve not reproduced at order %d', derN));
end
disp('testLegacyLine: PASS')
end
