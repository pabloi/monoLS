# monoLS
Monotonic function fittng to data.

#TO DO
Try to use lsqlin instead of lsqnonneg. Which is faster?
Separate the monotonic fitting (do assuming non-decreasing function with non-decreasing derivatives), from the logic used to determine which sign (non-decreasing or non-increasing) is best, and the logic to transform the data from any case to the non-decreasing case.
Add test scripts to show that the function(s) work with different parameter combinations.
Add example script to illustrate usage.
