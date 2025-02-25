import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_cov_ellipse(cov, pos, ax=None, **kwargs):
    """
    Following matplotlib documentation.
    """
    if ax is None:
        ax = init_cov_plot()

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    v = eigenvectors[:,0] # First eigenvector should be the largest.
    theta = np.degrees(np.arctan2(v[1], v[0]))

    # Takse square root to go from cov to std. Multiply because ellipse expects full width and height
    width, height = 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellipse)
    return ellipse, eigenvectors


def plot_2d_covariance_eigenvectors(Q, m, ax):
    """Arrows indiccating the principal componentes of a 2D cov matrix."""
    vals, vecs = np.linalg.eig(Q)
    vecs = vecs.T
    vals = np.sqrt(vals)

    for val, vi in zip(vals, vecs):
        plt.arrow(m[0], m[1], val*vi[0], val*vi[1], color='k', linewidth=2, head_width=0.05, length_includes_head=True)
    # plot_cov_ellipse(Q, m, nstd=1, ax=ax, alpha=0.1)
    return vals, vecs


def init_cov_plot():
    _, ax = plt.subplots(1,1, figsize=(3,3))
    ax.grid()
    ax.set_ylim((-1,1))
    ax.set_xlim((-1,1))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    return ax