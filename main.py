from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.regime_switching.markov_switching import cy_hamilton_filter_log, cy_kim_smoother_log
import warnings
import data
import numpy as np
from matrix_operations import vec_matrix, replace_diagonal
from scipy.linalg import sqrtm, pinv
from scipy import stats
from likelihood import SvarHMM
from matrix_operations import mat_vec

class HmmSvar(SvarHMM):
    def __init__(self, lags, regimes, x0, beta_hat=None,
                 **kwds):
        """
        :param endog:
        :param exog:
        :param initial_prob:
        :param transition_prob:
        :param lags:
        :param regimes:
        """
        self.theta_hat = None
        self.residuals = None
        self.sigma = None
        self.smoothed_marg_prob = None
        self.smoothed_joint_prob = None
        self.log_likelihoods = None
        self.log_eta = None
        self.initial_prob = None
        self.transition_prob = None
        self.endog = None
        self.exog = None
        self.lags = lags
        self.k_regimes = regimes
        self.beta_hat = beta_hat
        self.x = None
        self.order = 1
        self.k = None 
        self.bounds = None 
        self.resid_0 = None 
        self.sigma2 = None 
        self.log_eta0 = None 
        self.smoothed_prob0 = None   
    def start_params(self):
        """
        :param number_regimes: Number of regimes in the model
        :param no_lags:  Number of lags in the VECM estimates
        :param beta_hat: Default None, uses journalese procedure to tabulate
        :return:
        delta_y_t: endogenous variable
        z_t_1: right-hand variable
        ols_resid:  residuals from the initial OLS estimates of the VECM model
        """

        self.endog, self.exog, ols_resid = data.data_matrix(data.df, self.lags, self.beta_hat)
        k, obs = self.endog.shape
        self.k = k 

        # temp array to save u u.T values
        u_u = np.zeros([k * k, obs])
        # tabulate log squared residuals using the residuals
        for t in range(obs):
            u = ols_resid[:, t]
            u_u[:, t] = np.repeat(u, k) * np.tile(u, k)
        b_matrix = sqrtm(vec_matrix(u_u.sum(axis=1) / obs))
        b_matrix = b_matrix + np.random.normal(0.001, 2, size=(k, k))
        lam = replace_diagonal(np.random.normal(1, 0, size=k))
        sigma_array = np.zeros([self.k_regimes, k, k])
        for regime in range(self.k_regimes):
            if regime == 0:
                sigma_array[regime, :, :] = b_matrix @ b_matrix.T
            else:
                sigma_array[regime, :, :] = b_matrix @ lam @ b_matrix.T
        #TODO: Create OLS initial guesses. 
        shape_initial = k*k+ (self.k_regimes-1)*k
        initial_guess = np.ones(shape_initial)
        initial_guess[0:k*k] = mat_vec(b_matrix).squeeze()

        initial_params = {'regimes': self.k_regimes,
                          'epsilon_0': ((np.ones(self.k_regimes) / self.k_regimes)),
                          'transition_prob_mat': (np.ones([self.k_regimes, self.k_regimes]) / self.k_regimes),
                          'B_matrix': b_matrix,
                          'lambda_m': np.identity(b_matrix.shape[0]),
                          'sigma': sigma_array,
                          'residuals': ols_resid,
                          'VECM_params': None,
                          'x0': initial_guess  }
        return initial_params

    def fit(self, start_params=None, transformed=True,
            cov_kwds=None, maxiter=50, tolerance=1e-12, method='minimize', num_maxiter=1000):
        #print('grandchild fit called')
        if start_params is None:
            start_params = self.start_params()

            transformed = True
        else:
            start_params = np.array(start_params, ndmin=1)

        if not transformed:
            raise NotImplementedError
            # start_params = self.transform_params(start_params)
        #print('this is start params')
        #print(start_params)

        self.residuals = start_params['residuals']
        np.savetxt("ini_residuals.csv", self.residuals, delimiter=",")
        self.transition_prob = start_params['transition_prob_mat']
        np.savetxt("endog.csv", self.endog, delimiter=",") 
        np.savetxt("exog.csv",  self.exog, delimiter=",")
        self.theta_hat = None

        self.sigma = start_params['sigma']
        print(f'this is sigma{self.sigma}')
        self.x = start_params['x0']

        #print(f"initial prob {start_params['epsilon_0']}")

        self.initial_prob = start_params['epsilon_0']
        self.transition_prob = start_params['transition_prob_mat']

        # Perform expectation-maximization
        llf = []
        params = [start_params]
        i = 0
        delta = 0
        while i < maxiter and (i < 2 or (delta > tolerance)):

            self.expectation()
            if i == 0:
                self.smoothed_prob0 = self.smoothed_marg_prob

            print(f'=================this is itteration: {i} ====================')
            print('log eta t')
            print(self.log_eta)
            print('smoothed marginal probability')
            print(self.smoothed_marg_prob)
            print('=====================================')
            self.maximization(method, num_maxiter)
            llf.append(self.log_likelihoods.sum())
            np.savetxt("new_sigma", self.sigma[0,:,:])
            if i > 0:
                
                delta = 2 * (llf[-1] - llf[-2]) / np.abs((llf[-1] + llf[-2]))
            i += 1

    def expectation(self):
        # estimate eta t
        self.log_eta = self.cond_prob_(self.residuals)
        

        # filtered prob
        
        hamilton_result = cy_hamilton_filter_log(self.initial_prob,
                                                 self.transition_prob[..., np.newaxis], self.log_eta,
                                                 self.lags)
        #print(hamilton_result)
        predicted_joint_probabilities = hamilton_result[4]
        filtered_joint_probabilities = hamilton_result[5]
        self.log_likelihoods = hamilton_result[2]
        # smoothed prob
        kim_result = cy_kim_smoother_log(self.transition_prob[..., np.newaxis],
                                              predicted_joint_probabilities,
                                              filtered_joint_probabilities)
        self.smoothed_marg_prob = kim_result[1]
        for i in range(self.smoothed_marg_prob.shape[1]):
            if self.smoothed_marg_prob[:,i].sum() != 1:
                print('smoothed probability does not nadd up to 1') 
                self.smoothed_marg_prob[:,i] = self.smoothed_marg_prob[:,i] /  self.smoothed_marg_prob[:,i].sum()
        self.smoothed_joint_prob = kim_result[0]
        self.initial_prob = self.smoothed_marg_prob[:,0]

    def maximization(self, method, num_maxiter):
        # estimate transition probabilities
        self.transition_prob = self._em_regime_transition()
        self.get_bounds()
        # estimate B and Lambda matrices
        kwds =  {'min_method':'L-BFGS-B', 'bounds': self.bounds}
        optimization_class = SvarHMM(smoothed_prob=self.smoothed_marg_prob,
                        residuals=self.residuals,
                        start_params=self.x,  # start param
                        endog=self.endog,
                        exog=self.exog,
                        k_regimes=self.k_regimes,
                        loglike=None,
                        score=None,
                        hessian=None,
                        missing='none',
                        extra_params_names=None, **kwds)
        result = optimization_class.fit(method=method, maxiter=num_maxiter,**kwds)
        #print(f'this is result {result.summary()}')
        self.sigma = super().reconstitute_parameter_matrices(result.params, self.endog.shape[0])
        self.wls_res()
        

    def _em_regime_transition(self):
        """
        EM step for regime transition probabilities
        """

        # Marginalize the smoothed joint probabilities to just S_t, S_{t-1} | T
        tmp = self.smoothed_joint_prob 
        for i in range(tmp.ndim - 3):
            tmp = np.sum(tmp, -2)
        smoothed_joint_probabilities = tmp

        # Transition parameters (recall we're not yet supporting TVTP here)
        regime_transition = np.zeros((self.k_regimes, self.k_regimes))
        for i in range(self.k_regimes):  # S_{t_1}
            for j in range(self.k_regimes ):  # S_t
                regime_transition[i, j] = (
                                            np.sum(smoothed_joint_probabilities[j, i]) /
                                            np.sum(self.smoothed_marg_prob[i]))

            # It may be the case that due to rounding error this estimates
            # transition probabilities that sum to greater than one. If so,
            # re-scale the probabilities and warn the user that something
            # is not quite right
            delta = np.sum(regime_transition[i]) - 1
            if delta > 0:
                warnings.warn('Invalid regime transition probabilities'
                              ' estimated in EM iteration; probabilities have'
                              ' been re-scaled to continue estimation.', EstimationWarning)
                regime_transition[i] /= 1 + delta + 1e-6

        return regime_transition


    def cond_prob_(self, residuals):

        obs = residuals.shape[1]

        conditional_prob = np.zeros([self.k_regimes,self.order+1,  obs])
        for r in range(self.k_regimes):
            for j in range(self.order+1):
                for t in range(obs):
                    conditional_prob[r,j, t] = stats.multivariate_normal(mean=None, cov=self.sigma[r, :, :]).logpdf(residuals[:,t])
                    if t ==0:
                        print('-------------------------------------------------------')
                        print(f' this is conditional prob for t= 0 : {conditional_prob[r,j, t]}')
                        print(f'this is residuals: {residuals[:,t]}')
                        print( f'this is sigma {self.sigma[r, :, :]} ')
                        print('-------------------------------------------------------')
        return conditional_prob

    def wls_res(self):
        ####################################
        # estimate weighted least-square parameters
        ####################################
        k, obs = self.residuals.shape
        smoothed_prob = self.smoothed_marg_prob # TODO: changed smoothed marginal prob to inital marginal prob for testing the function 
        for regime in range(self.k_regimes):
            t_sum = np.zeros([self.exog.shape[0], self.exog.shape[0]])
            m_sum = np.zeros([self.exog.shape[0] * k, self.exog.shape[0] * k])
            m_sum_numo = np.zeros([self.exog.shape[0] * k, k])
            t_sum_numo = np.zeros([self.exog.shape[0] * k, 1])

            for t in range(obs):
                t_sum += smoothed_prob[regime, t] * self.exog[:, [t]] @ self.exog[:, [t]].T
            m_sum += np.kron(t_sum, pinv(self.sigma[regime, :, :]))
            denominator = pinv(m_sum)

        for t in range(obs):
            for regime in range(self.k_regimes):
                m_sum_numo += np.kron(smoothed_prob[regime, t] * self.exog[:, [t]], pinv(self.sigma[regime, :, :]))
            t_sum_numo += m_sum_numo @ self.endog[:, [t]]

        self.theta_hat = denominator @ t_sum_numo

        print(f'this is predicted theta hat {self.theta_hat}')


        resid = np.zeros(self.endog.shape)
        for t in range(obs):
            resid[:, [t]] = self.endog[:, [t]] - np.kron(self.exog[:, [t]].T, np.identity(k)) \
                                                                                @ self.theta_hat

        self.residuals = resid

    def get_bounds(self):
        if self.bounds == None:
            bound_list = []
            for i in range(len(self.x)):
                if i < self.k ** 2:
                    bound_list.append((None, None))
                else:
                    bound_list.append((0.01, None))
            self.bounds = tuple(bound_list)
        else:
            pass 


              

x0 = [5.96082486e+01, 5.74334765e-01, 2.83277325e-01, 3.66479528e+00,
      -2.08881529e-01, 6.32170541e-04, -1.09137417e-01, -3.80763529e-01,
      4.24379418e+00, 1.83658083e-01, 2.16692718e-03, 1.29590368e+00,
      2.20826553e+00, -2.98484217e-01, -5.38269363e-03, 1.19668239e-03,
      0.012, 0.102, 0.843, 16]

beta = np.array([0, 0, 0, 1]).reshape((-1, 1))
model1 = HmmSvar(3, 2, x0, beta_hat=beta)
model1.fit()