
# Translation Notes

The Bayes Generalized Linear Model (BGLM) is available in R through the `arm` [package](https://cran.r-project.org/web/packages/arm/arm.pdf) and was released with the corresponding paper titled [A Weakly Informative Default Prior Distribution for Logistic and other Regression Models](http://www.stat.columbia.edu/~gelman/research/published/priors11.pdf). BGLM is a Generalized Linear Model (GLM) that accepts prior mean, prior standard deviation, and prior degrees of freedom for the coefficients, and fits a model with additional pseudo-data points. While these values can be set, they also have out-of-the-box settings designed to generalized well with "weakly informative" default values. 

The BGLM model is available in R through the `arm` package through the `bayesglm` command which is built off of the standard `glm` implementation found in the R `stats` library. From the corresponding paper: 

> "We have implemented these computations by altering the `glm` function in R, creating a new function, `bayesglm`, that finds an approximate posterior mode and variance using extensions of the classical generalized linear model computations".

I was unable to find a Python package that implements the same BGLM algorithm as is found in the `arm` package. `statsmodels` does, however, contain a GLM model which was built to provide the same `glm` functionality in Python as is available in R. Since the `bayesglm` is built off the `glm` I used this as a starting point to translate the `bayesglm` process to Python. 

The process was to take the existing GLM from `statsmodels` which is accessed with
```python
from statsmodels.genmod.generalized_linear_model import GLM
```

And add an additional `method` argument, to solve the GLM with weakly informed priors. 

```python
model = BayesGLM(endog = endog, exog = exog, family = family)
result = model.fit() # New method here
```

Here I will break down the translation as it related to the paper as well as the `bayesglm` [codebase](https://github.com/suyusung/arm/blob/24bd6b3e1ebce5005b92c34a78f486908a436d37/R/bayesglm.R#L2) and the `bayesglm` [documentation in R](https://search.r-project.org/CRAN/refmans/arm/html/bayesglm.html). 

## Translation

I start with the `.fit()` method in `statsmodels`. The `.fit()` method itself only calls up one of the two private methods (`._fit_gradient()` or `._fit_irls()`) which is determined by the `method` argument which is passed to `.fit()`. The new `.fit()` method just has an additional `elif` statement to kick into the new `_fit_bayes_birls()` or "bayes iterative regularized least squares". 

The `_fit_bayes_irls()` is very similar to the `._fit_irls()`. The Gelman paper on page 1367: 
> "The standard logistic regression algorithm—upon which we build—proceeds by approximately linearizing the derivative of the log-likelihood, solving using weighted least squares, and then iterating this process, each step evaluating the derivatives at the latest estimate βˆ"

Therefore, I also begin by augmenting this process of solving using weighted least squares and iterating the process. Before the weighted least squares process begins, we need to implement the other novel component of BGLM: "pseudo-data and pseudo-variance values based on linearization of the derivative of the log-likelihood".

In standard weighted least squares, we would have some `X` covariates matrix, some `y` target vector, and some `W` weights matrix. However, we use the prior information passed to the model and create a few additional pseudo-data points. 

The steps to create these additional data points are outline in the paper start on page 1366, as well as in the R code. They begin with a logistic model:

$$Pr(y_i = 1) = logit^{-1}(X_i \beta) $$

Which is normally solved with maximum likelihood to get $\beta$ as a $\hat\beta$ estimate and a covariance matrix $V_\beta$ through the iterative least squares method mentioned earlier. 

However, the paper builds on this with: 

> "At each iteration, the algorithm determines pseudo-data $z_i$ and pseduo-variacnes $(\sigma_i^z)^2$ based on the linearization of the derivative of the log-likelihood."
> $$z_i = X_i\hat\beta + \frac{(1+e^{X_i\hat\beta})^2}{e^{X_i\hat\beta}}(y_i - \frac{e^{X_i\hat\beta}}{1+e^{X_i\hat\beta}})$$
> $$(\sigma_i^z)^2 = \frac{1}{n_i}\frac{(1+e^{X_i\hat\beta})^2}{e^{X_i\hat\beta}}$$
> "and then performs weighted least squares, regressing $z$ on $X$ with weight vector $(\sigma^z)^{-2}$. The resulting estimate $\hat\beta$ is used to update the computations [above], and the iteration proceeds until approximate convergence."

### z star
So here, we have $z_i$ and $(\sigma_i^z)^2$, where $z_i$ is in the code as `y` observations or `endog`.

In R we have 
```R
z <- (eta - offset)[good] + (y - mu)[good]/mu.eta.val[good]
```
We can break this down a bit. First, the `[good]` is just a filter for valid values, we can ignore this for now. The default `offset` value is also 0, so this can be ignored as well, leaving:
```R
z <- eta + (y - mu)/mu.eta.val
```
Which should correspond to our $z_i$ above. So we have 3 variables to look into further: `eta`, `mu` and `mu.eta.val` from R.

First, `mu` is created from `eta` with the line `mu <- linkinv(eta)` which comes from `linkinv <- family$linkinv`. So for the specific family link (binomial in the logit case). There's a link inverse. In the R codebase, there is a line that calls `(eval(family$initialize))` that gives starting values for `mu` with the equation `mustart <- (n * y + 0.5)/(n + 1)`. We can see this same setup in Python with `self.family.starting_mu(self.endog)` available here https://www.statsmodels.org/dev/generated/statsmodels.genmod.families.family.Binomial.starting_mu.html. 

Next, the `eta` term in R is calculate as `family$linkfun(mustart)`. So, again the family-specific link function (logit in our translation example), takes starting values and calculates this `eta`. In Python the setup uses `lin_pred = self.family.predict(mu)`. 

Lastly, `mu.eta.val` in R is calculated as `mu.eta.val <- mu.eta(eta)` where `mu.eta` is a "function: derivative of the inverse-link function with respect to the linear predictor". See: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/family.html. 


| R              | Python                       | Paper                                 | Paper Notation                                    | Notes                                                                                                                               |
| -------------- | ---------------------------- | ------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `eta`          | `lin_pred`                   | "linear predictor"                    | $X_i * \beta$                                     | `lin_pred = np.dot(self.exog, wls_results.params)`                                                                                  |
| `1/mu.eta.val` | `self.family.link.deriv(mu)` | "derivative of the logistic function" | $\frac{(1+e^{X_i\hat\beta})^2}{e^{X_i\hat\beta}}$ | Inverse of $\frac{\partial \mu}{\partial \eta}$.See [here](https://search.r-project.org/CRAN/refmans/LaplacesDemon/html/logit.html) |
| `mu`           | `mu`                         | "the expectation of y"                | $logistic(X_i*\beta)$                             |                                                                                                                                     |

```python
wlsendog = (
                lin_pred # eta in R, $X_i\hat\beta$ in the paper
                + self.family.link.deriv(mu) * (self.endog - mu)
                - self._offset_exposure # 0 by default
            )
```

So we now have the $z$ vector from the paper. This would be the same as standard GLM up to this point. However, according to equation (4) in the paper, we need to get $z_*$ by "...combining the likelihood [$z$ and $\sigma^z$, are the vectors of $z_i$'s and $\sigma_i^z$'s defined in (2)...] .. and the prior [$\mu$ and $\sigma$]..."

Therefore, $z_*$ is a vector comprised of the standard covariate inputs and a prior derived from equation 3 in the paper, given as:

>$$\beta_j \sim N(\mu_j,\sigma_j^2) \text{ for $j$ = 1, ..., $J$}$$

Therefore:

> $$z_* = \begin{pmatrix} z \\ \mu \end{pmatrix} $$

At the code level, we need to append $\mu$ to the $z_*$ vector. In R, this is set as `z.star <- c(z, prior.mean)` where prior mean is a constant argument included in the fit method. The constant is repeated to length `J` which is the number of features including intercept. In R `prior.mean <- rep(prior.mean, nvars)`. 

The Pythonic implementation then works the same by taking a constant mean and repeating it to the length of the covariates plus an intercept to achieve the `prior_mean` variable. The mean defaults to 0 as the paper states:
> As a default choice, we recommend the Cauchy distribution with center 0 and scale 2.5 

Giving `z_star = np.append(wlsendog, prior_mean)` as our first pseudo-data point. 

----
### X star

Next is the $X_*$ matrix of features with additional pseudo-data points, given in the paper as: 

> $$X_* = \begin{pmatrix} X \\ I_J \end{pmatrix} $$

Where X is the covariate inputs or exogenous variables, and $I_J$ is the "... $J \times J$ identity matrix ...", resulting in $X_*$ of shape $(n + J) \times J$.

In R, this is simply `x.star <- rbind(x, diag(NCOL(x)))` and in Python, this is:

```python
x_identity = np.identity(self.exog.shape[1])

x_star = np.vstack((self.exog, x_identity))
```
Where `self.exog` is just the covariate inputs (often X in scikit-like implementations). 

----
### w star

Lastly, is the weights matrix with additional pseudo-data points, given in the paper as:

> $$w_{*} = (\sigma^2, \sigma)^{-2} $$

Which is a vector of weights initialized using the link function.

In R, unless these are given to the function, then the weights are first initialized as all 1's: `weights <- rep.int(1, nobs)`  but then are passed through `w <- ((weights * mu.eta(eta)^2)/variance(mu))^0.5` where we again see the `mu.eta` and `mu` variables from before. 

In the Python implementation, the family-specific weights are taken from statsmodels using `self.weights = self.iweights * self.n_trials * self.family.weights(mu)`. Where `self.iweights` and `self.n_trials` are vectors of 1's for our implementation (the same as the R rep.int), so it's just multiplying those vectors by the family specific weight. See here for more details:
https://www.statsmodels.org/devel/generated/statsmodels.genmod.families.family.Family.weights.html

This gives us the $\sigma^2$ value. To get the $\sigma$ value, we use the `prior_scale` information, which by default is a value of 2.5. Which again comes from this part of the paper:

> As a default choice, we recommend the Cauchy distribution with center 0 and scale 2.5

This constant is repeated to a vector of size $J$ which is again the number of covariates.

Note: In both R and Python, while the `prior_scale` variable ends up creating a `prior_standard_deviation` or `prior.sd` variable which is repeatedly updated during the iteration through the least squares process. In R, we have `w.star <- c(w, sqrt(dispersion)/prior.sd)`. In Python we have `w_star = np.append(self.weights, np.sqrt(dispersion) / prior_standard_deviation)`. 

----

According to the Gelman paper, the advantage of setting up those pseudo-data points is:

> "With the augmented $X_∗$, this regression is identified, and, thus, the resulting estimate $\hat\beta$ is well defined and has finite variance, even if the original data have collinearity or separation that would result in nonidentifiability of the maximum likelihood estimate."


So from there we can implement this portion of the paper:

> "The full computation is then iteratively weighted least squares, starting with a guess of $\beta$ (for example, independent draws from the unit normal distribution), then computing the derivatives of the log-likelihood to compute $z$ and $\sigma_z$, then using weighted least squares on the pseudo-data (4) to yield an updated estimate of $\beta$, then recomputing the derivatives of the log-likelihood at this new value of $\beta$, and so forth, converging to the estimate $\hat\beta$"


Much of this is implemented already in GLM, since a standard GLM would do the same iterative weighted least squares (WLS) process. We can see this in the GLM codebase here:

```python
 wls_mod = reg_tools._MinimalWLS(wlsendog, wlsexog,
                                            self.weights, check_endog=True,
                                            check_weights=True)
wls_results = wls_mod.fit(method=wls_method)
```

Each iteration updates a `history` dictionary which has keys `"params"` and `"deviance"`. Once the `deviance` is minimized the model is considered converged.

The next line of the paper states that:

> "The covariance matrix $V_{\beta}$ is simply the inverse second derivative matrix of the log-posterior density evaluated at $\hat\beta$— that is, the usual normal-theory uncertainty estimate for an estimate not on the boundary of parameter space."

We need one additional step here in order to properly adapt this model. In the R package and in the paper, they are iteratively updating those prior pseudo-data points with results from each iteration of the WLS. Specifically, they create this $V_{\beta}$ covariance matrix and use it to update the `prior_standard_deviation` which is used in the $w_{*}$ vector as the pseudo-data component.

---
### Calculating $V_{\beta}$

In R, $V_{\beta}$ is calculated as follows:

```R
fit <- lm.fit(x = x.star[good.star, , drop = FALSE] * 
        w.star, y = z.star * w.star)

fit$qr$qr <- as.matrix(fit$qr$qr)

V.coefs <- chol2inv(fit$qr$qr[1:NCOL(x.star), 1:NCOL(x.star), 
        drop = FALSE])
```

I'll break this down, because it's difficult to decipher.

First, `lm.fit()` is calling this function: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm.fit
which is notably different from calling `lm()` and fitting a linear model object!

>`.lm.fit()` is bare bone wrapper to the innermost QR-based C code, on which glm.fit and lsfit are based as well, for even more experienced users.

Under the hood, `lm.fit()` is calling some C code that calls some FORTRAN code that solved the least squares through the QR decomposition matrix. Here's a blog that goes into deep detail: http://madrury.github.io/jekyll/update/statistics/2016/07/20/lm-in-R.html. 

For our purposes though, least squares is solved by QR matrix decomposition which states that the covariate matrix $X$ can be solved through matrix $QR$ using the Householders Reflections algorithm. More details on that can be found here: https://www.quantstart.com/articles/QR-Decomposition-with-Python-and-NumPy/. 

So from above, we call `lm.fit()` with our $X_*$ matrix. We are returned a fit object that stores the QR values as `fit$qr$qr <- as.matrix(fit$qr$qr)`. We need the `qr` values to calculate the `V.coefs` object. In R, the `fit$qr` object can be manipulated to show us 3 matrices: $X$, $Q$, and $R$ with `qr.X(fit$qr)`, `qr.Q(fit$qr)`, and `qr.R(fit$qr)` respectively. The object `fit$qr$qr` itself is actually the matrices $Q$ and $R$ merged together, specifically with the upper right triangle of $Q$ being replaced with the upper right triangle of $R$. Since $R$ can contain values in the upper right, but the lower triangle will always be 0's. Here's a quick example of what that looks like, but the details are really not important, because this specific implementation ONLY WORKS IN R:


$R$ matrix, note top right triangle are non-zero:
```R

Browse[6]> qr.R(fit$qr)
  (Intercept)          x1         x2
1   -4.331282 -0.04121481 -2.1656408
2    0.000000 -4.38171150  0.1974073
3    0.000000  0.00000000 -2.1928361
```

First 5 rows of the $Q$ matrix, note the upper right triangle
```R
Browse[6]> qr.Q(fit$qr)[1:5, 1:3]
            [,1]        [,2]        [,3]
[1,] -0.09997334 -0.09327594 -0.10713053
[2,] -0.09997334 -0.09240598  0.09041475
[3,] -0.09997334  0.11781638 -0.08812721
[4,] -0.09997334 -0.04556677 -0.10283557
[5,] -0.09997334  0.01824258  0.10037575
```

First 5 rows of the `fit$qr$qr`

Note the upper right triangle IS the $R$ matrix

Note the remainder of the matrix IS the $Q$ matrix
```R
Browse[6]> fit$qr$qr[1:5, 1:3]
  (Intercept)          x1          x2
1 -4.33128157 -0.04121481 -2.16564078
2  0.09997334 -4.38171150  0.19740731
3  0.09997334 -0.12629395 -2.19283614
4  0.09997334  0.03708919  0.09652572
5  0.09997334 -0.02672016 -0.11258139
```

In R, a subset of the `fit$qr$qr` matrix is used to create the `V.coef` variables as:

```R
V.coefs <- chol2inv(fit$qr$qr[1:NCOL(x.star), 1:NCOL(x.star), 
        drop = FALSE])
```

Where we take the cholskey inverse of that `fit$qr$` matrix using the `chol2inv` function, which we don't have in Python. Luckily, checking the docs it can easily be calculated: https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/chol2inv.

`chol2inv` is equivalent to calculating $(R'R)^{-1}$. So in Python, we just take the R component of the QR matrix decomposition. Then we calculate R Transpose times R and take the inverse of the result. 

This QR stuff is computation gymnastics for all OLS. [See this post for the math](https://math.stackexchange.com/a/687361). But in essence, this is a way to calculate the variance of the coefficients for all OLS models. This is true regardless of the solving method, since the result of OLS is deterministic.

The use of the R matrix also seems to be either a convenience or computation thing, because you can get the same (np.close) answer with the weighted covariate matrixVar_with_X = np.linalg.inv(np.dot(wls_mod.wexog.T, wls_mod.wexog)).

The last step with the $V_\beta$ matrix is to updated the `prior_standard_deviation` variable and use it to updated the pseudo-data for the weights vector. In R: 

```R
 prior.sd <- ifelse(prior.df == Inf, prior.scale, 
        sqrt(((coefs.hat - prior.mean)^2 + diag(V.coefs) * 
          dispersion + prior.df * prior.scale^2)/(1 + 
          prior.df)))
```

Pythonically:
```python
if self.prior_df == np.Infinity:
    prior_standard_deviation = prior_scale.copy()
else:
    prior_standard_deviation = np.sqrt(
        (
            ((wls_results.params - self.prior_mean) ** 2)
            + np.diag(v_coefs) * dispersion
            + self.prior_df * prior_scale**2
        )
        / (prior_df + 1)
    )
```

More detail on this is given in equation (8) in the Gelman paper:

> $$ \hat\sigma^2 = \frac{(\hat\beta_j - \mu_j)^2 + (V_\beta)_{jj} + v_js_j^2}{1+v_j}$$

> which corresponds to the (approximate) posterior mode of σj2 given a single
measurement with value (7) and an $Inv-\chi^2 (v_j , s_j^2 )$ prior distribution.

The paper contains more details on this exact implementation. For this above equation (8), the code is clear as to where this is implemented. Equations 5 and 6 are not as clear though, and are noted as being implemented only when the "coefficients on $\beta_j$ have independent $t$ prior distributions with centers $\mu_j$ and scales $s_j$..."

The rest of the algorithm continues as the GLM does normally. The last bit of the new method just implements the standard glm results wrapper.

## Scope of this Translation
What will separate this BGLM implementation from standard GLM is both in the details in the solver (above), but also in our added flexibility for prior inputs to the model. One notable difference between R and Python flexibility is that we have not (as of writing) implemented all bells and whistles from the BGLM R package. Namely, these are the arguments for R's bayesglm.fit:

```R
function (x, y, weights = rep(1, nobs), start = NULL, etastart = NULL, 
mustart = NULL, offset = rep(0, nobs), family = gaussian(), 
control = list(), intercept = TRUE, prior.mean = 0, prior.scale = NULL, prior.df = 1, prior.mean.for.intercept = 0, prior.scale.for.intercept = NULL, prior.df.for.intercept = 1, min.prior.scale = 1e-12, scaled = TRUE, 
print.unnormalized.log.posterior = FALSE, Warning = TRUE)
```

For our model, we currently only retain:

```python
    def _fit_bayes_irls(
        self,
        start_params=None,
        maxiter: int = 100,
        tol: float = 1e-8,
        cov_type: str = "nonrobust",
        cov_kwds=None,
        use_t=None,
        weights: np.array = None,
        perform_scale: bool = True,
        prior_mean: float = 0,
        prior_scale: float = DEFAULT,
        prior_df: float = 1,
        prior_mean_for_intercept: float = 0,
        prior_scale_for_intercept: float = DEFAULT,
        prior_df_for_intercept: float = 1,
        **kwargs)
```
Which we believe is a faithful representation of the core functionality. As an example, we do implement a `prior_scale` functionality but currently don't support a `min.prior.scale`