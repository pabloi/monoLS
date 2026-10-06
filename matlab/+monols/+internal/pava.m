function z = pava(y, w)
%PAVA Weighted pool-adjacent-violators: the non-decreasing weighted LS fit, O(n).
n = numel(y);
level = zeros(n, 1); weight = zeros(n, 1); count = zeros(n, 1);
b = 0;
for i = 1:n
    b = b + 1;
    level(b) = y(i); weight(b) = w(i); count(b) = 1;
    while b > 1 && level(b-1) > level(b)
        wsum = weight(b-1) + weight(b);
        level(b-1) = (weight(b-1)*level(b-1) + weight(b)*level(b)) / wsum;
        weight(b-1) = wsum;
        count(b-1) = count(b-1) + count(b);
        b = b - 1;
    end
end
z = zeros(n, 1);
pos = 0;
for i = 1:b
    z(pos+1:pos+count(i)) = level(i);
    pos = pos + count(i);
end
end
