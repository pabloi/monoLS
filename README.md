# monoLS
Data smoothing through monotonic curve fitting for Matlab and Octave. It is a nonparametric approach to smoothing/curve-fitting that is useful when we know that the data must be increasing or decreasing, but don't have additional information about it or don't want to use additional assumptions. 

`monoLS` finds the optimal monotonic (i.e. non-negative or non-positive derivative) curve fit to some dataset of (x,y) points. Higher-order derivatives can also be constrained to be of constant sign (or 0), but the same sign has to be used for all odd-order derivatives and for all even-order derivatives.
A regularization term is used to avoid undue deference to datapoints near the extremes of the x range.
By default, `monoLS` uses mean-squared errors (2-norm) so it is the least-square solution. In this case, the problem can be framed as a non-negative least squares problem (which is always convex). Any p-norm with finite non-zero p can be used too.

## Problem solving
The large family of problems described above (increasing or decreasing best-fits, with either concave or convex curves and forced signs for the first N-derivatives provided that all odd and all even derivatives have the same sign) can be reduced to the problem of finding a best-fitting function to some data where the first N derivatives are non-negative.
This is done by flipping as needed the y-axis (which flips the sign of all the derivatives), and the x-axis sign (which flips the sign of odd derivatives only). 

The problem of finding a solution to the best-fitting function with non-negative first N derivatives can be framed as a p-norm minimization problem on a convex (non-negative) set. It can be shown that this is a convex problem, and thus can be . For the special case of the 2-norm, this is a non-negative least-squares problem. In general, the problem can be cast as:

$$min_w || Aw - y ||_p \, \, \text{s.t.} \, \, w_i \geq 0, \,\, \forall i$$

Where `A` is a triangular matrix, `z=Aw` are the smoothed values we are searching for, and most of the `w_i` represent the n-th order differentials (i.e. the value of the n-th derivative at the sampling points).

Currently, `monoLS` is able to enforce non-negativity (or non-positivity) up to the 3rd derivative of the data. For forcing up to 2nd derivative, it uses the lsqnonneg routine as a solver. For forcing up to the 3rd derivative, it uses quadprog which is slower but is better behaved numerically. It fails to converge for higher order derivatives although a global optimum must exist (the problem is convex, although numerically ill-conditioned).

## Basic syntax

## Requirements
**Matlab:** Optimization toolbox required.

**Octave:** optim, struct, statistics, and io packages required.

For both, the monoLS folder needs to be added to the path.

## Code structure:
The code contains two folders: `fun` and `examples`.  

**`fun` folder:** contains the `incLS` (numeric solver), `monoLS` (wrapper of incLS for additional functionality), and `monoLS2` (experimental alternative solver that does not use `incLS`, no longer supported).

**`examples` folder:** contains three test scripts illustrating use and results of `monoLS`.