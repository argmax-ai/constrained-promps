import numpy as np
import scipy as sp
import scipy.linalg

from opt_pmp_utils.basis_functions import GaussianRBF


class ProMP(object):
    """Basic ProMP class

    Can fit to data with EM method.

    Parameters:
        number_of_outputs: output dimensionality
        basis_fun: desired basis function class (typically GaussianRBF)
        std_y: measurement noise standard deviation
        std_prior: Prior standard deviation
        robust: enables robust learning of the ProMP
        fit_noise: enables fitting measurement noise
    """

    def __init__(
        self,
        number_of_outputs,
        basis_fun=GaussianRBF(
            20,
            np.linspace(0, 1, 101),
            std_distance=1.0,
            normalize_features=True,
            c_t_delta=0.1,
        ),
        std_y=0.001,
        std_prior=100.0,
        robust=False,
        fit_noise=False,
    ):
        self.basis_fun = basis_fun
        self.n_basis = self.basis_fun.n_basis
        self.n_obs = number_of_outputs
        self.std_y = std_y
        self.std_prior = std_prior
        self.robust = robust
        self.fit_noise = fit_noise
        self._generate_features()

    def _generate_features(self):
        # Construct feature matrix
        self.X = self.basis_fun.X
        self.X_stacked = np.kron(np.eye(self.n_obs), self.X[:, np.newaxis, :])

        self.m0 = np.zeros(self.n_basis * self.n_obs, np.float)
        self.V0 = np.eye(self.n_basis * self.n_obs, dtype=np.float) * (
            self.std_prior**2
        )
        self.Q0 = np.eye(self.n_basis * self.n_obs, dtype=np.float) / (
            self.std_prior**2
        )
        self.Vy = np.eye(self.n_obs, dtype=np.float) * (self.std_y**2)
        self.Qy = np.eye(self.n_obs, dtype=np.float) / (self.std_y**2)

    def _extract_block_diag(self, A, blength):
        multiples = A.shape[0] / blength
        if np.floor(multiples) != np.ceil(multiples):
            raise RuntimeError(
                "Matrix cannot be split evenly into blocks of the given blocklength"
            )
        A_bd_ones = sp.linalg.block_diag(
            *[np.ones([blength, blength]) for _ in range(int(multiples))]
        )
        return A * A_bd_ones

    def _sp_spd_inv(self, A):
        L_A = sp.linalg.cholesky(A, lower=True)
        rhs = sp.linalg.solve_triangular(L_A, np.eye(A.shape[0]), lower=True)
        A_inv = sp.linalg.solve_triangular(L_A.T, rhs)
        return 0.5 * (A_inv + A_inv.T)

    def e_step(self, y):
        Q = np.kron(self.Qy, self.X.transpose() @ self.X) + self.Q0
        Q = 0.5 * (Q + Q.T)
        V = self._sp_spd_inv(Q)
        rhs = self.Q0 @ self.m0 + np.ravel(self.X.transpose() @ y @ self.Qy, order="F")
        m = V @ rhs
        if self.fit_noise:
            M = np.transpose(np.reshape(m, [self.n_obs, self.n_basis]))
            error = y - self.X @ M
            sig_y_sq = np.einsum("to,to->o", error, error) + np.einsum(
                "tob,bj,toj->o", self.X_stacked, self.V0, self.X_stacked
            )
            self.sig_y_sq += sig_y_sq
        return m, V

    def m_step(self, m_list, V_list):
        n_i = len(m_list)
        assert n_i == len(V_list)
        m = np.average(m_list, axis=0)
        m0 = np.reshape(m, [-1, 1])
        V = []
        for m_i, V_i in zip(m_list, V_list):
            m_i = m_i[..., np.newaxis]
            V.append(((m_i - m0) @ (m_i - m0).transpose() + V_i))
        V = np.average(V, axis=0)
        if self.robust:
            V_block_diag = self._extract_block_diag(V, self.n_basis)
            N0 = 2 * (self.n_basis * self.n_obs + 1)
            V = 1 / (n_i + N0) * (n_i * V + N0 * V_block_diag)
        if self.fit_noise:
            self.sig_y_sq = self.sig_y_sq / n_i / self.basis_fun.T
        return m, V

    def EM(self, Y):
        m_list = []
        V_list = []
        if self.fit_noise:
            self.sig_y_sq = 0
        for y in Y:
            m_i, V_i = self.e_step(y)
            m_list.append(m_i)
            V_list.append(V_i)
        m0, V0 = self.m_step(m_list, V_list)
        return m0, V0

    def fit(self, Y, em_iter):
        self.n_iter_em = em_iter
        for i in range(self.n_iter_em):
            m, V = self.EM(Y)
            self.m0 = m
            self.V0 = V
            self.Q0 = self._sp_spd_inv(self.V0)
            if self.fit_noise:
                self.Vy = np.diag(self.sig_y_sq)
                self.Qy = np.diag(1 / self.sig_y_sq)

        self.condition = np.linalg.cond(self.V0)

    def project(self):
        # Final values
        self.M = np.transpose(np.reshape(self.m0, [self.n_obs, self.n_basis]))
        # Posterior predictive distribution
        self.X_s = np.kron(np.eye(self.n_obs), self.X)
        self.my_pp = self.X @ self.M
        self.myt = np.einsum("tw,w->t", self.X_s, self.m0)
        # Final y variance
        self.Vyt = np.einsum("mb,bj,nj->mn", self.X_s, self.V0, self.X_s)
        # Compute Vy_pp via einsum
        self.Vy_pp = np.einsum(
            "tob,bj,tij->toi", self.X_stacked, self.V0, self.X_stacked
        )

    def reset(self):
        self._generate_features()

    def add_via_point(self, point, cov, xt):
        xt_stacked = np.kron(np.eye(self.n_obs), xt)
        K = (
            self.V0
            @ xt_stacked.T
            @ self._sp_spd_inv(cov + xt_stacked @ self.V0 @ xt_stacked.T)
        )
        self.m0 = self.m0 + K @ (point - xt_stacked @ self.m0)
        self.V0 = self.V0 - K @ xt_stacked @ self.V0

    def plot_covariance_at_timeidx(self, timeidx):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 1)
        Vy_diag = np.sqrt(np.diag(self.Vy_pp[timeidx, :, :])[:, np.newaxis])
        pos = axs.imshow(self.Vy_pp[timeidx, :, :] / (Vy_diag @ Vy_diag.T))
        fig.colorbar(pos, ax=axs)
        axs.set_title("Covariance at timeidx {}".format(timeidx))

    def sample(self, number_of_samples):
        w_samples = np.random.multivariate_normal(
            self.m0, self.V0, size=number_of_samples
        )
        w_samples = np.transpose(
            w_samples.reshape([number_of_samples, self.n_obs, self.n_basis]), [0, 2, 1]
        )
        y_samples = np.einsum("tb,sbo->sto", self.X, w_samples)
        return y_samples


def test_promp():
    import matplotlib.pyplot as plt

    from opt_pmp_utils.planarRobot import ControlledDualPlanarRobot

    np.random.seed(77)
    robust = False
    n_basis = 30
    n_iter_em = 10
    nLinks = 2
    nLinks_total = 2 * nLinks
    nSamples = 100
    pGain = 20
    originA = [-2, 0]
    originB = [2, 0]
    linkLength = 2
    robot = ControlledDualPlanarRobot(
        pGain=pGain,
        dGain=0.8 * 2 * np.sqrt(pGain),
        noise=2.0,
        nLinks=nLinks,
        linkLengthA=np.ones(2) * linkLength,
        linkLengthB=np.ones(2) * linkLength,
        originA=originA,
        originB=originB,
    )

    q0 = np.zeros(nLinks * 4)  # [q0A, dq0A, q0B, dq0B]
    q0[0] = np.pi / 2
    q0[nLinks_total] = np.pi / 2
    tEnd = 1.0
    dt = 0.01
    tVec = np.array([0.0, tEnd])
    rVec = np.zeros([tVec.size, 2 * nLinks_total])
    final_target_distance = 0.4
    target_angle = np.arcsin(
        (np.abs(originA[0] - originB[0]) - final_target_distance) / 4 / linkLength
    )
    rVec[:, :nLinks] = np.ones([tVec.size, nLinks]) * [
        np.pi / 2 - target_angle,
        -np.pi + (2 * target_angle),
    ]
    rVec[:, nLinks_total : nLinks_total + nLinks] = np.ones([tVec.size, nLinks]) * [
        np.pi / 2 + target_angle,
        np.pi - (2 * target_angle),
    ]
    tEval = np.linspace(0, tEnd, num=int(tEnd // dt + 1))
    mean, cov = robot.evolveMeanCov(q0, tVec, rVec, tEval)
    tSamples, samples = robot.sample(nSamples, q0, tVec, rVec, tEnd, dt)

    samplesA, samplesB = robot._split_q(samples)
    samplesA = samplesA[:, :, :nLinks]
    samplesB = samplesB[:, :, :nLinks]
    bf = GaussianRBF(
        n_basis,
        time_vec=tSamples,
        std_distance=1.0,
        normalize_features=True,
        c_t_delta=0.1,
    )
    pmpA = ProMP(nLinks, bf)
    pmpB = ProMP(nLinks, bf)
    pmpA.fit(samplesA, n_iter_em)
    pmpB.fit(samplesB, n_iter_em)
    pmpA.project()
    pmpB.project()

    fig, axs = plt.subplots(nLinks, 2, figsize=(10, 10))
    for j in range(2):
        for i in range(nLinks):
            idx = i + j * nLinks_total
            didx = idx + nLinks

            # Plot position
            axs[i, j].plot(tEval, mean[:, idx], "b", label="mean")
            axs[i, j].fill_between(
                tEval,
                mean[:, idx] - 3 * np.sqrt(cov[:, idx, idx]),
                mean[:, idx] + 3 * np.sqrt(cov[:, idx, idx]),
                color="b",
                alpha=0.2,
                # label="variance",
            )
            art_s = axs[i, j].plot(
                tSamples, samples[:, :, idx].transpose(), "k", alpha=0.3
            )
            art_s[0].set_label("samples")
            if j == 0:
                axs[i, j].set_title("qA{}".format(i))
                mean_pmp = pmpA.my_pp[:, i]
                cov_pmp = pmpA.Vy_pp[:, i, i]
            else:
                axs[i, j].set_title("qB{}".format(i))
                mean_pmp = pmpB.my_pp[:, i]
                cov_pmp = pmpB.Vy_pp[:, i, i]

            # Plot promp
            axs[i, j].plot(tEval, mean_pmp, "r", label="promp")
            axs[i, j].fill_between(
                tEval,
                mean_pmp - 3 * np.sqrt(cov_pmp),
                mean_pmp + 3 * np.sqrt(cov_pmp),
                color="r",
                alpha=0.2,
                # label="variance",
            )
            axs[i, j].plot(
                tSamples,
                np.mean(samples[:, :, idx], axis=0),
                "g",
                linestyle="--",
                alpha=1.0,
                label="sample mean",
            )
            leg = axs[i, j].legend()
            for l in leg.get_lines():
                l.set_alpha(1)
    plt.show()


if __name__ == "__main__":
    test_promp()
