import numpy as np


class GaussianRBF(object):
    """GaussianRBF basis functions for ProMP research

    Contains only __init__ function which computes feature matrix and
    corresponding derivatives.

    Parameters:
        n_basis: Number of basis functions
        time_vec: one-dim vector of timepoints
        std_distance: modifies the standard deviation of the kernels,
            1.0 = sigma is exactly the distance between centers
            2.0 = sigma is twice the distance between centers, and so on
        normalize_features: True = sum of basis functions at each timepoint is 1
        c_t_delta: allows basis function centers outside of the given time range
    """

    def __init__(
        self,
        n_basis: int,
        time_vec: np.ndarray = np.linspace(0, 1, num=101),
        std_distance: float = 1.0,
        normalize_features: bool = True,
        c_t_delta: float = 0.0,
    ):
        super(GaussianRBF, self).__init__()
        self.n_basis = n_basis
        self.time_vec = time_vec
        self.std_distance = std_distance
        self.normalize_features = normalize_features
        self.c_t_delta = c_t_delta

        self.t = self.time_vec.reshape(-1, 1)
        self.T = self.t.size
        self.dt = np.gradient(self.time_vec)

        self.center = np.linspace(
            self.t[0] - self.c_t_delta, self.t[-1] + self.c_t_delta, num=self.n_basis
        ).reshape((-1, 1))
        self.sig_b = np.abs(self.center[1] - self.center[0]) / self.std_distance
        self.dX2 = (
            np.matmul(self.t**2, np.ones((1, self.n_basis)))
            - 2 * np.matmul(self.t, self.center.transpose())
            + np.matmul(np.ones((self.T, 1)), self.center.transpose() ** 2)
        )

        self.X = np.exp(-0.5 * self.dX2 / (self.sig_b**2))
        if self.normalize_features:
            self.X = self.X / np.sum(self.X, axis=1)[..., np.newaxis]

        self.dX = (
            self.X
            * (
                np.ones([self.T, 1]) @ self.center.transpose()
                - self.t @ np.ones([1, self.n_basis])
            )
            / (self.sig_b**2),
        )
        self.ddX = self.X * (self.dX2 - (self.sig_b**2)) / (self.sig_b**4)


def main():
    n_basis = 20
    bf = GaussianRBF(n_basis)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    for i in range(n_basis):
        ax.plot(bf.time_vec, bf.X[:, i])
    plt.show()


if __name__ == "__main__":
    main()
