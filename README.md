# SuperGaussian

We provide a *superfast* set of tools for parametric inference in stationary Gaussian processes.

Parametric likelihoods for multi-dimensional time series typically consist of stationary Gaussian processes, for which 
traditional *fast* inference algorithms scale as O(N^2) in the number of observations. Here we present and implement a *superfast* 
O(N log^2 N) algorithm for parametric inference for stationary Gaussian processes, including novel superfast
algorithms for score and Hessian calculations.
