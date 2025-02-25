__author__ = 'henk'
"""Some elementary kalman functions."""
import numpy as np
from numpy.linalg import inv
from numpy import dot
import matplotlib.pyplot as plt


def predict(x, cov_x, a_transition, cov_transition, u=None):
    """ Textbook one-step kalman prediction.

    Args:
        x: (x_prev) state vector (n elements)
        cov_x: (P_prev) covariance matrix of x. P=E{xx'}  (n by n elements)
        a_transition: (A) Transition matrix (n by n elements)
        cov_transition: (Q) Uncertainty added by the transition.
        u: Control input.

    Returns:
        x: State verctor at new time.
        P: covariance matrix associated with new x.

        The following equation is implemented:
        x = A*x_prev + u
        P = Pprev * A * Pprev' + Q
    """
    x = dot(a_transition, x)
    if u is not None:
        x = x + u
    cov_x = dot(a_transition, dot(cov_x, a_transition.T)) + cov_transition
    return x, cov_x


def update(x, cov_x, y, h_measurement, cov_measurement):
    """ Textbook one-step measurement update.

    """
    e = y - h_measurement@x                                   # Innovation
    s = cov_measurement + h_measurement@cov_x@h_measurement.T # Covariance of innovation
    kalman_gain = cov_x@h_measurement.T@inv(s)
    x = x + kalman_gain@e
    # P = P - dot(K, dot(S, K.T))  # Standard form

    # Prevent numerical instability
    # See also http://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
    A = np.eye(x.shape[0]) - dot(kalman_gain, h_measurement)
    cov_x = dot(dot(A, cov_x), A.T) + dot(dot(kalman_gain, cov_measurement), kalman_gain.T)
    cov_x = 0.5 * (cov_x + cov_x.T)
    return x, cov_x


def update_with_innovation_cov(x, cov_x, y, h_measurement, cov_measurement):
    """ Textbook one-step measurement update.
    
        Same as 'update' function, but also outputs inv(S), the innovations matrix. This can be used to 
        check the consistency of the updates. (does the uncertainty match the predicted uncertainty)

    """
    e = y - h_measurement@x                                   # Innovation
    s = cov_measurement + h_measurement@cov_x@h_measurement.T # Covariance of innovation
    inv_s = inv(s)
    kalman_gain = cov_x@h_measurement.T@inv_s
    x = x + kalman_gain@e
    # P = P - dot(K, dot(S, K.T))  # Standard form

    # Prevent numerical instability
    # See also http://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
    A = np.eye(x.shape[0]) - dot(kalman_gain, h_measurement)
    cov_x = dot(dot(A, cov_x), A.T) + dot(dot(kalman_gain, cov_measurement), kalman_gain.T)
    cov_x = 0.5 * (cov_x + cov_x.T)
    return x, cov_x, inv_s


def extract_sd_from_cov_matrix(cov, n=0, m=None):
    if m is None:
        return np.sqrt(np.diag(cov))
    else:
        return np.sqrt(np.diag(cov[n:m, n:m]))


def correlation_matrix(cov, indx=None, triangular_only=True, do_plot=False):
    """ Given a covariance matrix, return the correlation P[i,j]/sd[i]/sd[j]
        I selects what channels to return (default all)"""
    if indx is None:
        indx = list(range(cov.shape[0]))
    nI = len(indx)
    C = np.zeros((nI, nI))
    for i in range(nI):
        for j in range(nI):
            i_P, j_P = indx[i], indx[j]
            if i == j:
                C[i, j] = np.sqrt(cov[i_P, i_P])
            else:
                C[i, j] = cov[i_P, j_P] / np.sqrt(cov[i_P, i_P] * cov[j_P, j_P])
                # if i == 0:
                #     print('------' + str(i_P) + ',' + str(j_P))
                #     print(P[i,j])
    if triangular_only:
        C = np.triu(C)
        C[C == 0] = None

    if do_plot:
        plt.imshow(C, cmap='gist_yarg', interpolation='nearest')
    return C


def check_covariance_positive_definite(cov, verbose=False):
    """ Actually check diagonal only."""
    I: list = np.where(np.diag(cov) <= 0)[0]
    if verbose:
        print('Negative values for index: {}'.format(I))
    if len(I) == 0:
        return True
    else:
        return False


def add_matrix_diagonally(a, b):
    """Extend/concatenate matrix in diagonal direction.

    Args:
        a: 2D numpy array in left upper corner
        b: 2D numpy array in right lower corner.

    Returns:
        Concatenation accoording to [A,   zero ]
                                    [zero,    B]

    """
    z = np.zeros((b.shape[0], a.shape[1]))  # Bottom filling matrix
    # np.bmat([[A,Z],[Z.T,B]])
    # return np.bmat([[A,Z.T], [Z, B]])
    return np.block([[a, z.T], [z, b]])
