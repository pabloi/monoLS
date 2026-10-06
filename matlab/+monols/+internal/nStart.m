function ns = nStart(N, k)
%NSTART Number of start coefficients (intercept included) for N samples and order k.
ns = min(k, N - 1) + 1;
end
