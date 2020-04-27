# Superfast Computation Involving Toeplitz Matrices

With class **Toeplitz** we can achieve the *superfast* computation speed when computing the multiplication or inversion of Toeplitz matrices. Here is the example:

``` cpp
int N;                                               /// size of Toeplitz matrix
double* V_acf;                                       /// first column of the Toeplitz matrix
double* x;                                           /// an array of length N
double* y;                                           /// an array of length N for storing the results

Toeplitz* Tz = new Toeplitz(N);
Tz->setAcf(V_acf);
Tz->mult(y, x);                                       /// y = V * x
Tz->solve(y, x);                                      /// y = V^-1 * x
double ldet = Tz->logDet();                           /// log determinant of V
```


# Superfast Inference for Stationary Gaussian Models

Consider a length $N$ stationary Gaussian process $\bm X \mid \bm\theta = [x_1, \ldots, x_N]$ with mean $0$ and covariance $\bm V_{\bm\theta}, \bm V_{\bm\theta}[i, j] = \text{cov}(x_i, x_j)\mid\bm\theta$:

$$
\bm X \sim \bm N(\bm 0, \bm V_{\bm\theta})
$$

where $\bm\theta = [\theta_1,\ldots, \theta_p]$ is the collection of unknown parameters. The likelihood of such time series model is

$$
\ell(\bm\theta \mid \bm X) = -\frac{1}{2}[\bm X' V_\theta \bm X + |V_\theta|]
$$

In various statistical applications, the evaluation of the likelihood $\ell(\bm\theta \mid \bm X)$, together with its gradient $\frac{\partial}{\partial \theta_i}\ell(\bm\theta \mid \bm X)$ and Hessian matrix $\frac{\partial^2}{\partial \theta_i \partial \theta_j}\ell(\bm\theta \mid \bm X)$ are repeatedly required. With class **NormalToeplitz** we can compute these statistics in *superfast* speed. Here is the example:

``` cpp
int N;                                              /// length of time series
int p;                                              /// number of parameters
double* X;                                          /// length-N array of the time series
double* acf;                                        /// length-N array of the first column of the covariance matrix
double* dX;                                         /// length-(N*p) array of the derivatives of X w.r.t p different parameters
double* dacf;                                       /// length-(N*p) array of the derivatives of acf w.r.t p different parameters
double* d2X;                                        /// length-(N*p*p) array of the second order derivatives of X w.r.t p different parameters
double* d2acf;                                      /// length-(N*p*p) array of the second order derivatives of acf w.r.t p different parameters
double* dl;                                         /// length-p array that stores the gradients (Score function)
double* d2l;                                        /// length-(p*p) array that stores the Hessian matrix

NormalToeplitz* Nt = new NormalToeplitz(N, p);
double ldens = Nt->(x, acf);                        /// log density, aka log-likelihood
Nt->grad(dl, X, dX, acf, dacf);                     /// dl is the gradients
Nt->hess(dl, X, dX, d2X, acf, dacf, d2acf);         /// d2l is the Hessian matrix
```