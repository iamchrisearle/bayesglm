# bayesglm.py

# This file is part of BayesGLM.
#
# BayesGLM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# any later version.
#
# BayesGLM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesGLM. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import statsmodels.api as sm
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod.generalized_linear_model import (
    GLM,
    GLMResults,
    GLMResultsWrapper,
)
from statsmodels.tools.sm_exceptions import PerfectSeparationError

DEFAULT = float()

model_families = {
    "gaussian": sm.families.Gaussian(),
    "logit": sm.families.Binomial(),  # Logit by default
    "probit": sm.families.Binomial(link=sm.genmod.families.links.Probit()),
    "poisson": sm.families.Poisson(),
}


def _check_convergence(criterion, iteration, atol, rtol):
    return np.allclose(
        criterion[iteration], criterion[iteration + 1], atol=atol, rtol=rtol
    )


class BayesGLM(GLM):
    def __init__(self, *args, **kwargs):
        if "family" in kwargs and isinstance(kwargs["family"], str):
            kwargs["family"] = model_families[kwargs["family"]]

        super().__init__(*args, **kwargs)

    def fit(
        self,
        start_params=None,
        maxiter=100,
        tol=1e-8,
        scale=None,
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=None,
        prior_mean: float = 0,
        prior_scale: float = DEFAULT,
        prior_df: float = 1,
        prior_mean_for_intercept: float = 0,
        prior_scale_for_intercept: float = DEFAULT,
        prior_df_for_intercept: float = 1,
        **kwargs,
    ):
        """
        Fits a generalized linear model for a given family.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is family-specific and is given by the
            ``family.starting_mu(endog)``. If start_params is given then the
            initial mean will be calculated as ``np.dot(exog, start_params)``.
        maxiter : int, optional
            Default is 100.
        tol : float
            Convergence tolerance.  Default is 1e-8.
        scale : str or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default value is None, which uses `X2` for Gamma, Gaussian,
            and Inverse Gaussian.
            `X2` is Pearson's chi-square divided by `df_resid`.
            The default is 1 for the Binomial and Poisson families.
            `dev` is the deviance divided by df_resid
        cov_type : str
            The type of parameter estimate covariance matrix to compute.
        cov_kwds : dict-like
            Extra arguments for calculating the covariance of the parameter
            estimates.
        use_t : bool
            If True, the Student t-distribution is used for inference.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        atol : float, optional
            The absolute tolerance criterion that
            must be satisfied. Defaults to ``tol``. Convergence is attained
            when: :math:`rtol * prior + atol > abs(current - prior)`
        rtol : float, optional
            The relative tolerance criterion that
            must be satisfied. Defaults to 0 which means ``rtol`` is not used.
            Convergence is attained when:
            :math:`rtol * prior + atol > abs(current - prior)`
        tol_criterion : str, optional
            Defaults to ``'deviance'``. Can
            optionally be ``'params'``.
        optim_hessian : {'eim', 'oim'}, optional
            (available with scipy optimizer fits) When 'oim'--the default--the
            observed Hessian is used in fitting. 'eim' is the expected Hessian.
            This may provide more stable fits, but adds assumption that the
            Hessian is correctly specified.
        """
        if isinstance(scale, str):
            scale = scale.lower()
            if scale not in ("x2", "dev"):
                raise ValueError("scale must be either X2 or dev when a string.")
        elif scale is not None:
            # GH-6627
            try:
                scale = float(scale)
            except Exception as exc:
                raise type(exc)("scale must be a float if given and no a string.")
        self.scaletype = scale

        return self._fit_bayes_irls(
            start_params=start_params,
            maxiter=maxiter,
            tol=tol,
            scale=scale,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
            prior_mean=prior_mean,
            prior_scale=prior_scale,
            prior_df=prior_df,
            prior_mean_for_intercept=prior_mean_for_intercept,
            prior_scale_for_intercept=prior_scale_for_intercept,
            prior_df_for_intercept=prior_df_for_intercept,
            **kwargs,
        )

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
        **kwargs,
    ):
        """
        Fits a Bayes generalized linear model for a given family using
        iteratively reweighted least squares (IRLS). See Gelman (2008)
        http://www.stat.columbia.edu/~gelman/research/published/priors11.pdf
        """
        attach_wls = kwargs.pop("attach_wls", False)
        # sets up tolerances for convergence check see:
        # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        atol = kwargs.get("atol")
        rtol = kwargs.get("rtol", 0.0)
        tol_criterion = kwargs.get("tol_criterion", "deviance")
        wls_method = kwargs.get("wls_method", "lstsq")
        atol = tol if atol is None else atol
        endog = self.endog
        wlsexog = self.exog

        # ==== vars for BGLM ====
        # Based on statsmodel's doc:
        # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html  # noqa
        # "An intercept is not included by default and should be added by the user."
        # An intercept is triggered by the exog's first col being a constant
        # self.k_constant == 0 for no intercept, and self.k_constant == 1 for intercept
        intercept = self.k_constant > 0
        nvars = wlsexog.shape[1]
        # Set default values
        default_probit_scale_factor = (
            1.6 if isinstance(self.family.link, sm.families.links.Probit) else 1
        )
        if prior_scale is DEFAULT:
            prior_scale = 2.5 * default_probit_scale_factor
        if prior_scale_for_intercept is DEFAULT:
            prior_scale_for_intercept = 10 * default_probit_scale_factor
        # Set start priors
        if intercept is True:
            prior_scales = np.array(
                [prior_scale_for_intercept] + [prior_scale] * (nvars - 1)
            )
            prior_means = np.array(
                [prior_mean_for_intercept] + [prior_mean] * (nvars - 1)
            )
            prior_dfs = np.array([prior_df_for_intercept] + [prior_df] * (nvars - 1))
        else:
            prior_scales = np.array([prior_scale] * nvars)
            prior_means = np.array([prior_mean] * nvars)
            prior_dfs = np.array([prior_df] * nvars)

        # In R: scaled = True
        if perform_scale is True:
            # scales the prior_scales, which is then appended to the w* vector as sigma
            if isinstance(self.family, sm.families.Gaussian):
                prior_scales = prior_scales * 2 * self.endog.std(ddof=1)
                # retain the original prior to use later
                prior_scales_original = prior_scales.copy()
            for i in range(nvars):
                x_obs = self.exog[:, i]
                num_categories = len(np.unique(x_obs))
                if num_categories == 2:
                    x_scale = x_obs.max() - x_obs.min()
                elif num_categories > 2:
                    x_scale = 2 * x_obs.std(ddof=1)
                else:
                    x_scale = 1
                prior_scales[i] = prior_scales[i] / x_scale

        # this gets used to update w* later on
        prior_standard_deviations = prior_scales.copy()

        # In R we have 3 starting params: start, etastart, and mustart
        # In R we have: mustart <- (weights * y + 0.5)/(weights + 1)
        # Should be what eval() does in R
        if start_params is None:
            start_params = np.zeros(nvars)
            # Starting mu is just mui = yi +.5/(2)
            # https://www.statsmodels.org/dev/generated/statsmodels.genmod.families.family.Binomial.starting_mu.html
            mu = self.family.starting_mu(y=self.endog)
            # lin_pred here returns exact same values
            # as self.family.link(mu) which returns exact same thing
            # as in R where we call family$linkfun(mustart)
            # so this IS `eta` in R.
            # This is backed up by the documentation here:
            # https://www.statsmodels.org/dev/generated/statsmodels.genmod.families.family.Binomial.predict.html
            # that says that self.family.predict() returns:
            # `lin_pred`` is a `ndarray`
            # Linear predictors based on the mean response variables.
            # The value of the link function at the given mu.
            lin_pred = self.family.predict(mu)
        else:
            lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
            mu = self.family.fitted(lin_pred)
        # This is a default scale depending on family
        # It's 1 for Binomials, Negative Binomials, and Poisson
        # Else it's Pearson's chi-squared estimate
        self.scale = self.estimate_scale(mu)
        # This should match the dev.resids in R
        # which similarily comes from a family attribute
        # Starts as devold and then they update it in R.
        dev = self.family.deviance(
            self.endog, mu, self.var_weights, self.freq_weights, self.scale
        )
        if np.isnan(dev):
            raise ValueError(
                "The first guess on the deviance function "
                "returned a nan.  This could be a boundary "
                " problem and should be reported."
            )

        # first guess on the deviance is assumed to be scaled by 1.
        # params are none to start, so they line up with the deviance
        history = dict(params=[np.inf, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history[tol_criterion]

        # For weights, in R we have
        # w.star <- c(w, sqrt(dispersion)/prior.sd)
        # so we need dispersion and prior.sd
        # dispersion is always 1 for Binomial and Poisson
        if isinstance(self.family, sm.families.Binomial) or isinstance(
            self.family, sm.families.Poisson
        ):
            dispersion = 1
        else:
            dispersion = np.var(self.endog) / 10000
        # This special case is used to get the likelihood for a specific
        # params vector.
        if maxiter == 0:
            mu = self.family.fitted(lin_pred)
            self.scale = self.estimate_scale(mu)
            wls_results = lm.RegressionResults(self, start_params, None)
            iteration = 0
        for iteration in range(maxiter):
            # Normal glm
            weights = self.iweights * self.n_trials * self.family.weights(mu)
            weights_square_rooted = np.sqrt(weights)
            wlsendog = (
                lin_pred
                + self.family.link.deriv(mu) * (self.endog - mu)
                - self._offset_exposure
            )
            # ==== data augmentation ====
            # From the Gelman paper:
            # Xstar = (X
            #          Ij)
            # So just create the identity matrix and stack it
            # to get shape n+J x J
            x_identity = np.identity(nvars)
            if perform_scale and intercept:
                x_identity[0] = np.mean(self.exog, axis=0)

            x_star = np.vstack((self.exog, x_identity))
            z_star = np.append(wlsendog, prior_means)
            # Weights are squared inside of the WLS.fit step in python,
            # but outside in R.
            # So our w_star values are squared compared to the R values.
            w_star = np.append(
                weights, dispersion / np.square(prior_standard_deviations)
            )
            # ==== end of data augmentation ====

            wls_mod = reg_tools._MinimalWLS(
                endog=z_star,
                exog=x_star,
                weights=w_star,
                check_endog=True,
                check_weights=True,
            )
            # Same as GLM
            wls_results = wls_mod.fit(method=wls_method)
            lin_pred = np.dot(self.exog, wls_results.params)
            lin_pred += self._offset_exposure
            mu = self.family.fitted(lin_pred)
            history = self._update_history(wls_results, mu, history)
            self.scale = self.estimate_scale(mu)
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)

            # ==== update the priors/the EM step in Gelman et al. 2008  ====
            # Based on _MinimalWLS' fit implementation using the qr method
            Q, R = np.linalg.qr(wls_mod.wexog)
            # In R: V.coefs <- chol2inv(fit$qr$qr[1:NCOL(x.star), 1:NCOL(x.star), drop = FALSE])  # noqa
            # chol2inv's docs say "Equivalently, compute (X'X)^-1 from the (R part) of
            # the QR decomposition of X." X it the R matrix from QR decomp.
            # There is no existing comparable method to chol2inv in Python.
            v_coefs = np.linalg.inv(np.dot(R.T, R))

            if isinstance(self.family, sm.families.Gaussian) and perform_scale is True:
                prior_scales = prior_scales_original.copy()

            # In R:
            # prior.sd <- ifelse(prior.df == Inf, prior.scale,
            #            sqrt(((coefs.hat - prior.mean)^2 + diag(V.coefs) *
            #            dispersion + prior.df * prior.scale^2)/(1 +
            #            prior.df)))
            if prior_df == np.Infinity:
                prior_standard_deviations = prior_scales.copy()
            else:
                prior_standard_deviations = np.sqrt(
                    (
                        ((wls_results.params - prior_means) ** 2)
                        + np.diag(v_coefs) * dispersion
                        + prior_dfs * prior_scales**2
                    )
                    / (prior_dfs + 1)
                )

            # In R:
            # mse.resid <- mean((w * (z - x %*% coefs.hat))^2)
            # mse.uncertainty <- mean(rowSums((x %*% V.coefs) *
            # x)) * dispersion
            # dispersion <- mse.resid + mse.uncertainty
            if not (
                isinstance(self.family, sm.families.Binomial)
                or isinstance(self.family, sm.families.Poisson)
            ):
                mse_resid = np.mean(
                    (
                        weights_square_rooted
                        * (wlsendog - np.dot(self.exog, wls_results.params))
                    )
                    ** 2
                )
                mse_uncertainty = (
                    np.mean(np.sum(np.dot(self.exog, v_coefs) * self.exog, axis=1))
                    * dispersion
                )
                dispersion = mse_resid + mse_uncertainty

            converged = _check_convergence(criterion, iteration + 1, atol, rtol)
            if converged:
                break
        self.mu = mu

        # Running lm.WLS.fit does not change the resulting params
        # It simply calculates additional results, e.g. normalized_cov_params
        if maxiter > 0:  # Only if iterative used
            wls_method2 = "pinv" if wls_method == "lstsq" else wls_method
            wls_model = lm.WLS(z_star, x_star, w_star)
            wls_results = wls_model.fit(method=wls_method2)

        glm_results = GLMResults(
            self,
            wls_results.params,
            wls_results.normalized_cov_params,
            self.scale,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
        )

        glm_results.method = "BIRLS"
        glm_results.mle_settings = {}
        glm_results.mle_settings["wls_method"] = wls_method
        glm_results.mle_settings["optimizer"] = glm_results.method
        if (maxiter > 0) and (attach_wls is True):
            glm_results.results_wls = wls_results
        history["iteration"] = iteration + 1
        glm_results.fit_history = history
        glm_results.converged = converged
        return GLMResultsWrapper(glm_results)
