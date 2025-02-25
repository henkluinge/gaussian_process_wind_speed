import numpy as np

def evaluate_fourier_series(x, a, b):
    """ 
    Standard  fourier evaluation on all values of x. Variable names according to 
    https://math.mit.edu/~gs/cse/websections/cse41.pdf

    b[0] is not used.
    """
    s = a[0]
    for n, (ai, bi) in enumerate(zip(a[1:],b[1:]), 1):
        s += ai*np.cos(n*x) + bi*np.sin(n*x)
    return s


def integrate_fourier_coeficients(df, K=6, n_integration_bins=360, channel='wspd'):
    """ 
    Integrate specifically with the wind direction model: 
    - Data is not equally sampled.
    - Interval is from 0 to 2pi.

    K is the number of fourier coefficeints to compute.

    Since data is not equally sampled, we define an equal set of intervals. Then integrated over all points within that interval.
    """
    x = np.linspace(0, 2*np.pi, n_integration_bins, endpoint=True)
    K = 6
    a = np.zeros(K)
    b = np.zeros(K)
    for i in range(len(x)-1):
        mask= (df.index>=x[i]) & (df.index<x[i+1])
        x_mid = (x[i] + x[i+1])/2
        x_int = x[i+1] - x[i]
        m = df[mask][channel].mean(skipna=True)
        if m != m:
            m = m_prev

        a[0] += m*x_int

        for k in range(1, K):
            a[k] += m*np.cos(k*x_mid)*x_int
            b[k] += m*np.sin(k*x_mid)*x_int

        m_prev = m
    a = a/np.pi
    b = b/np.pi
    a[0] = a[0]/2
    return a, b