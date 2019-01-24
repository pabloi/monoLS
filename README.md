# monoLS
Data smoothing through monotonic curve fitting for Matlab and Octave.

MonoLS finds the optimal monotonic (i.e. non-negative or non-positive derivative) curve fit to some dataset of (x,y) points. Higher-order derivatives can also be constrained to be of constant sign (or 0), but the same sign has to be used for all odd-order derivatives and for all even-order derivatives.
A regularization term is used to avoid undue deference to datapoints near the extremes of the x range.
By default, monoLS uses mean-squared errors so it is the least-square solution. In this case, the problem can be framed as a non-negative least squares problem (which is always convex).

Currently, monoLS is able to enforce non-negativity (or non-positivity) up to the 3rd derivative of the data. For forcing up to 2nd derivative, it uses the lsqnonneg routine as a solver. For forcing up to the 3rd derivative, it uses quadprog which is slower but is better behaved numerically. It fails to converge for higher order derivatives although a global optimum must exist (the problem is convex, although numerically ill-conditioned).

MonoLS works in Matlab (toolboxes required?) and Octave, provided that the optim package for Octave is installed (which in turn depends on the struct, statistics, and io packages).

#TO DO
Add test scripts to show that the function(s) work with different parameter combinations.
Add example script to illustrate usage.
Fix convergence issues for high-order derivatives being constrained.
Optimize for speed, check that global optimum is achieved.
Modify so that function is octave-ready (currently works with octave if the optim package is installed, but need to disable optimoptions construction)
