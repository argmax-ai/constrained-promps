import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot2dNormal(m, V, ax, conf=0.95, color="k", fc="None", linestyle="-", alpha=1.0):
    eigvals, eigvecs = np.linalg.eig(V)
    eigidx = np.argsort(eigvals)

    confVal = chi2.ppf(conf, 2)
    ax.add_artist(
        Ellipse(
            m,
            2 * np.sqrt(confVal * eigvals[eigidx[0]]),
            2 * np.sqrt(confVal * eigvals[eigidx[1]]),
            angle=180
            / np.pi
            * np.arctan(eigvecs[1, eigidx[0]] / eigvecs[0, eigidx[0]]),
            fc=fc,
            ec=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=alpha,
        )
    )
    ax.scatter(m[0], m[1], c=color, marker="X", s=50)
