from statsmodels.base.model import GenericLikelihoodModel
import numpy as np


class SvarHMM(GenericLikelihoodModel):
    def __int__(self, smoothed_prob,residuals,start_params, endog, exog, k_regimes=2, loglike=None,
                score=None, hessian=None,
                missing='none', extra_params_names=None, **kwds):

        super(SvarHMM, self).__init__(endog=endog, exog=exog, loglike=loglike, score=score,
                                      hessian=hessian, missing=missing,
                                      extra_params_names=extra_params_names, kwds=kwds)
        self.residuals = residuals
        self.smoothed_prob = smoothed_prob
        self.y = np.array(self.endog)
        self.k_regimes = k_regimes

        self.mu_matrix = []

        # The vector of initial values for all the parameters, beta and q, that the optimizer will
        # optimize
        self.start_params = start_params

        print('self.start_params=' + str(self.start_params))
        # A very tiny number (machine specific). Used by the LL function.
        self.EPS = np.finfo(np.float64).eps
        # Optimization iteration counter
        self.iter_num = 0


    def nloglikeobs(self, params):
        #Reconstitute the q and beta matrices from the current values of all the params
        self.reconstitute_parameter_matrices(params) #TODO: need IMplementation
        #Let's increment the iteration count
        self.iter_num= self.iter_num+1
        # Compute all the log-likelihood values for the Poisson Markov model
        ll = self.compute_loglikelihood() #TODO: need implimentation
        #Print out the iteration summary
        print('ITER='+str(self.iter_num) + ' ll='+str(((–ll).sum(0))))
        #Return the negated array of  log-likelihood values
        return –ll