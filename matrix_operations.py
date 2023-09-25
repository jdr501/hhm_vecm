import numpy as np

from scipy.special import logsumexp

def mat_vec(matrix):
    """
    This function vectorized any matrix
    :param matrix: should be an ndarray
    :return: returns a vector of the matrix as numpy (row*Col, ) array
    """
    return matrix.T.flatten().reshape((-1, 1))


def vec_matrix(vec, r=None, c=None):
    """
    :param vec: numpy array
    :param r: rows of the matrix
    :param c: columns of the matrix
    :return: numpy ndarray
    """
    if r is None and c is None:
        r = c = int(np.sqrt(len(vec)))
    if r is None and type(c) == int:
        r = len(vec) - c
    if c is None and type(r) == int:
        c = len(vec) - r
    return vec.reshape(-1, 1).reshape(c, r).T


def resid(dy_t, z_t, theta, ik=None):
    if ik is None:
        ik = np.identity(dy_t.shape[0])
    u_hat = np.zeros(dy_t.shape)
    for i in range(dy_t.shape[1]):
        z = z_t[:, [i]]
        dy = dy_t[:, [i]]
        #print(f'this is the shape of z: {z.shape}')
        # print(f'this is the shape of dy: {dy.shape}')
        # print(f'this is the shape of theta: {theta.shape}')
        u_hat[:, [i]] = dy - np.kron(z.T, ik) @ theta

    return u_hat


def vec_summation(epsilon_t_t_upper_m, u_hat):
    vec_sum = 0
    for t in range(len(epsilon_t_t_upper_m)):
        residuals = u_hat[:, [t]]
        vec_sum = vec_sum + epsilon_t_t_upper_m[t] * (residuals @ residuals.T)
    return vec_sum


def replace_diagonal(replacement_list):
    """
    :param replacement_list: array of numbers
    :return: identity matrix with array of numbers
    """
    matrix = np.identity(len(replacement_list))
    for i in range(len(replacement_list)):
        matrix[i, i] = replacement_list[i]
    return matrix


def log_space_product(A,B):
    Astack = np.stack([A]*B.shape[1]).transpose(1,0,2)
    Bstack = np.stack([B]*A.shape[0]).transpose(0,2,1)
    return logsumexp(Astack+Bstack, axis=2)