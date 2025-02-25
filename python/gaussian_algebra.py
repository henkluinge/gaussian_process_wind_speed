""" Routines for textbook gaussian manipulations. Used in wind_direction_model.ipynb
"""
import numpy as np


# def gaussian_condition(m, Q, x2):
#     """ Data:
#         x = [x1
#              x2]  

#         Q = [A   B] 
#             [B.T C]

#         x2 is known (data)

#         Computes p(x1 | x2)

#         Blunt textbook implementation. For large systems use kalman recursion or cholesky decomposition.
#     """
#     n = len(m) - len(x2)
#     m1 = m[:n]
#     m2 = m[n:]
#     A =  Q[:n, :n]
#     B =  Q[:n, n:]
#     C =  Q[n:, n:]
#     Cinv = np.linalg.inv(C)

#     m1 = m1 + B@Cinv@(x2-m2)
#     Q1 = A - B@Cinv@B.T
#     return m1, Q1


def gaussian_condition(m, Q, x1, sigma=1):
    """ Data:
        x = [x1
             x2]  

        Q = [A   B] 
            [B.T C]

        x1 is known (data)

        Computes p(x2 | x1)

        Blunt textbook implementation. For large systems use kalman recursion or cholesky decomposition.
    """
    n = len(x1)
    m1 = m[:n]
    m2 = m[n:]
    A =  Q[:n, :n]
    B =  Q[:n, n:]
    C =  Q[n:, n:]
    s_sq = sigma**2
    Inoise = s_sq*np.eye(n)
    Ainv = np.linalg.inv(A + Inoise)

    m2 = m2 + B.T@Ainv@(x1-m1)
    Q2 = C - B.T@Ainv@B
    return m2, Q2

def gaussian_marginalize_lower(m, Q, n):
    """ Given a gaussian with mean vector m and covariance matrix Q, select just the values with index > n.

        Partitioning:
        m = [m1
             m2]  

        Q = [A   B] 
            [B.T C]

        Returns m1, A
    """
    m_marginalized = m[n:]
    Q_marginalized = Q[n:, n:]
    return m_marginalized, Q_marginalized


def gaussian_marginalize_upper(m, Q, n):
    """ Given a gaussian with mean vector m and covariance matrix Q, select just the first n values.

        Partitioning:
        m = [m1
             m2]  

        Q = [A   B] 
            [B.T C]

        Returns m1, A
    """
    m_marginalized = m[:n]
    Q_marginalized = Q[:n, :n]
    return m_marginalized, Q_marginalized


def get_2D_cov_matrix(angle_deg=20, scale=(1, 0.5)):
    """ Generate cov matrix for testing and plotting. 
    """
    m = np.asarray((0.,0.))
    Q = np.eye(len(m))
    # Scale then rotate
    alpha = angle_deg*np.pi/180
    R = np.asarray([[np.cos(alpha), np.sin(-alpha)], [np.sin(alpha), np.cos(alpha)]])
    S = np.diag(scale)
    T = R@S  # Transformation matrix
    Q = T@Q@T.T
    return m, Q


def K_rbf(x, l=1):
    """Radial basis function kernel"""
    delta_x = x[1]-x[0]
    r_sq = delta_x**2
    l_sq = l**2
    return np.exp(-r_sq/(2*l_sq))


def K_periodic_simple(x, l=2.3):
    """Periodic on exactly 2pi"""
    delta_x = x[1]-x[0]
    num = -2*np.sin(delta_x/2)**2
    l_sq = l**2
    # return np.exp(num/(l_sq))
    return 1+ num/2

def K_periodic(x, l=2.3):
    """Periodic on exactly 2pi"""
    delta_x = x[1]-x[0]
    num = -2*np.sin(delta_x/2)**2
    l_sq = l**2
    return np.exp(num/(l_sq))
    # return 1+ num/2




if __name__ == '__main__':

    # Try out functions.
    m = np.asarray((0.,0., 0.))
    Q = np.eye(len(m))
    Q[1,0], Q[0, 1] = 0.7, 0.7
    Q[2,0], Q[0, 2] = 0.7, 0.7
    Q[1,2], Q[2, 1] = 0.7, 0.7
    x2 = np.asarray((1.0,))
    print('Test variables')
    print('m', m)
    print('x2', x2)
    print('Q', Q)
    
    print('-- Conditioning--')
    print(gaussian_condition(m, Q, x2))

    if False:
        print('-- Marginalization --')
        m_marginalized, Q_marginalized = gaussian_marginalize_upper(m, Q, n=2)
        print(m_marginalized, Q_marginalized)
