import pandas as pd
from statsmodels.tsa.vector_ar import vecm
import numpy as np

# import data
df = pd.read_csv('slowdown.csv')
# changing year to a readable quarterly date
df['year'] = pd.date_range("1970-01-01", freq="Q", periods=len(df)).to_period('Q')
# select the variables we need
df = df.loc[:, ['year', 'OIL', 'YUS', 'CPUS', 'SUS']]
df['YUS'] = 100 * df['YUS']
df['CPUS'] = 100 * df['CPUS']
# Take subset of data to match the paper
df = df[df['year'] >= pd.Period('1980Q1')]
df.set_index('year', inplace=True)


# VECM Model Fit
def data_matrix(data_frame, lag, beta=None):
    """
    :param data_frame: Data set
    :param lag: number of lags in the model
    :param beta: specify theory driven betas or Default uses Johansen's procedure to estimate beta
    :return:  delta_y_t- left-hand side matrix  , z_t_1 right-hand side matrix
    """
    model = vecm.VECM(endog=data_frame,
                      k_ar_diff=lag,
                      coint_rank=1,
                      dates=data_frame.index,
                      deterministic="colo")
    fitted = model.fit()
    resid = fitted.resid.T  # residual shape is (variables, observations)
    if beta is None:
        beta = fitted.beta  # shape of beta is (number of variables, 1)
        print(f'this is the original shape {beta.shape}')

    data_mat = vecm._endog_matrices(model.y, model.endog, model.exog, lag, "colo")
    k, obs = data_mat[0].shape
    delta_y_t = data_mat[1]  # left-hand variables
    beta_trn_y_t_1 = np.array(
        [(beta.T @ data_mat[2][:, i].T)
         for i in range(obs)]).T
    delta_y_t_lags = data_mat[3][: -(k + 2)]
    v0_v1 = data_mat[3][-(k + 2):- k]
    z_t_1 = np.vstack((v0_v1, beta_trn_y_t_1, delta_y_t_lags))
    return delta_y_t, z_t_1, resid
