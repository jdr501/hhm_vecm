from __future__ import annotations
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
from matrix_operations import vec_matrix, replace_diagonal
from statsmodels.tools.numdiff import approx_hess1, approx_hess2, approx_hess3
from scipy import stats






class SvarHMM(GenericLikelihoodModel):

    def __init__(self, smoothed_prob, residuals, start_params, endog, exog, k_regimes, loglike=None,
                 score=None, hessian=None,
                 missing='none', extra_params_names=None, **kwds):
        super(SvarHMM, self).__init__(endog=endog.T, exog=exog.T, loglike=loglike, score=score,
                                      hessian=hessian, missing=missing,
                                      extra_params_names=extra_params_names, kwds=kwds)
        self.residuals = residuals
        self.smoothed_prob = smoothed_prob
        self.y = np.array(self.endog)
        self.k_regimes = k_regimes
        self.iter_num = 0
        self.mu_matrix = []

        # The vector of initial values for all the parameters, beta and q, that the optimizer will
        # optimize
        self.start_params = start_params

        print('self.start_params=' + str(self.start_params))
        # A very tiny number (machine specific). Used by the LL function.
        self.EPS = np.finfo(np.float64).eps
        # Optimization iteration counter

    def nloglikeobs(self, params):
        # Reconstitute the q and beta matrices from the current values of all the params
        sigma = self.reconstitute_parameter_matrices(params,self.endog.shape[1])
        # Let's increment the iteration count
        self.iter_num = self.iter_num + 1
        # Compute all the log-likelihood values for the Poisson Markov model
        ll = self.compute_loglikelihood(sigma)
        # Print out the iteration summary
        # Return the negated array of  log-likelihood values
        return -ll

    def reconstitute_parameter_matrices(self, params,endog_shape):
        regimes = self.k_regimes
        k = endog_shape 

        b_mat = vec_matrix(np.array(params[0:k ** 2]))
        lam_m = np.zeros([regimes - 1, k, k])
        start = k * k
        for m in range(regimes - 1):
            end = start + k
            lam_m[m, :, :] = replace_diagonal(params[start:end])
            start = end

        sigma = np.zeros([regimes, k, k])
        for regime in range(regimes):
            if regime == 0:
                sigma[regime, :, :] = b_mat @ b_mat.T
            else:
                sigma[regime, :, :] = b_mat @ lam_m[regime - 1, :, :] @ b_mat.T

        return sigma

    def compute_loglikelihood(self, sigma):
        likelihood_array = np.zeros([self.k_regimes, self.residuals.shape[1]])
        for regime in range(self.k_regimes):
            likelihood_array[regime, :] = (self.smoothed_prob[regime, :] * stats.multivariate_normal(
                mean=None, cov=sigma[regime, :, :], allow_singular=True).logpdf(self.residuals.T).T)
        return likelihood_array.sum(axis=0).sum()

    # This function just tries its best to compute an invertible Hessian so that the standard
    # errors and confidence intervals of all the trained parameters can be computed successfully.
    def hessian(self, params):
        for approx_hess_func in [approx_hess3, approx_hess2, approx_hess1]:
            H = approx_hess_func(x=params, f=self.loglike, epsilon=self.EPS)
            if np.linalg.cond(H) < 1 / self.EPS:
                print('Found invertible hessian using ' + str(approx_hess_func))
                return H
        print('DID NOT find invertible hessian')
        H[H == 0.0] = self.EPS
        return H
