import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from opt_pmp_utils.basis_functions import GaussianRBF
from opt_pmp_utils.utils import sp_spd_inv, tf_vectorize


class CProMP(object):
    """Constrained ProMPs class

    Create object from class, use `add_constraint` and `add_penalty` to create your problem.
    Use the `fit` function to run the cpmp algorithm.

    Parameters:
        number_of_outputs: output dimensionality
        basis_fun: desired basis function class (typically GaussianRBF)
        const: list of constraint objects
        penalty: list of penalty objects
    """

    def __init__(
        self,
        number_of_outputs,
        basis_fun=GaussianRBF(20, np.linspace(0, 1, 101)),
        const=[],  # Constraints
        penalty=[],  # Penalties
        floatT=tf.float64,
    ):
        self.n_obs = number_of_outputs
        self.basis_fun = basis_fun
        self.n_basis = basis_fun.n_basis
        self.n_weights = self.n_basis * self.n_obs
        self.const = const
        self.penalty = penalty
        self.floatT = floatT
        self.lagrange_updates = 0
        self.kl_scale = 1.0
        self.kl_no_corl = False
        self.kl_n_blocks = 1
        self._generate_features()

    def add_constraint(self, constraint):
        self.const.append(constraint)

        self.n_t_const = 0
        self.c_scatter_idx = []
        for c in self.const:
            c_size = tf.size(c.lagrange_var)
            self.c_scatter_idx.append(tf.range(c_size)[:, tf.newaxis] + self.n_t_const)
            self.n_t_const += c_size.numpy()
        self.c_list = tf.zeros(self.n_t_const, dtype=self.floatT)

    def add_penalty(self, new_penalty):
        self.penalty.append(new_penalty)

    def _generate_features(self):
        # Extract features
        self.X = tf.constant(self.basis_fun.X, dtype=self.floatT)
        self.dX = tf.constant(self.basis_fun.dX, dtype=self.floatT)
        self.ddX = tf.constant(self.basis_fun.ddX, dtype=self.floatT)

    def _get_lagrange_list(self):
        # Return a list which includes all the lagrange variables
        lagrange_list = []
        for c in self.const:
            lagrange_list.append(c.lagrange_var)
        return lagrange_list

    def tf_fun_graph(self, M, log_diag_L_V, off_diag_L_V):
        lagrange_list = self._get_lagrange_list()
        loss, c_list, kl, p_sum = self._tf_fun_graph(
            M, log_diag_L_V, off_diag_L_V, lagrange_list
        )
        self.c_list = c_list
        self.kl_tmp = kl
        self.p_sum_tmp = p_sum
        return loss

    def graph(self, M, log_diag_L_V, off_diag_L_V):
        lagrange_list = self._get_lagrange_list()
        loss, c_list, kl, p_sum = self._graph(
            M, log_diag_L_V, off_diag_L_V, lagrange_list
        )
        self.c_list = c_list
        self.kl_tmp = kl
        self.p_sum_tmp = p_sum
        return loss

    @tf.function
    def _tf_fun_graph(self, M, log_diag_L_V, off_diag_L_V, lagrange_list):
        return self._graph(M, log_diag_L_V, off_diag_L_V, lagrange_list)

    def _graph(self, M, log_diag_L_V, off_diag_L_V, lagrange_list):
        # The actual computation graph
        self.M = M
        self.M_vec = tf_vectorize(M, name="M_vec")
        self.L_V_tmp = tfp.math.fill_triangular(off_diag_L_V)
        self.L_V = self.L_V_tmp + tf.linalg.tensor_diag(
            -tf.linalg.diag_part(self.L_V_tmp) + tf.exp(log_diag_L_V)
        )
        self.V = self.L_V @ tf.transpose(self.L_V)
        self.w_dist = tfp.distributions.MultivariateNormalTriL(
            loc=self.M_vec, scale_tril=self.L_V
        )

        self.p_sum = tf.constant(0.0, dtype=self.floatT)
        for p in self.penalty:
            self.p_sum += tf.reduce_sum(p.evaluate(self.w_dist))

        self.my_pp = self.X @ M
        self.X_lop = tf.linalg.LinearOperatorFullMatrix(
            self.X[:, tf.newaxis, :], name="LinearOperatorFullMatrix_X"
        )
        self.X_stacked = tf.linalg.LinearOperatorKronecker(
            [
                tf.linalg.LinearOperatorIdentity(
                    num_rows=self.n_obs, dtype=self.floatT
                ),
                self.X_lop,
            ]
        ).to_dense()
        self.Vy_pp = tf.einsum(
            "tob,bj,tij->toi", self.X_stacked, self.V, self.X_stacked
        )

        self.marginals = tfp.distributions.MultivariateNormalFullCovariance(
            loc=self.my_pp,
            covariance_matrix=self.Vy_pp,
            name="marginals",
        )

        # KL between two multivariate gaussians
        self.mean_diff = tf.subtract(self.M_vec, self.m0, name="mean_diff")
        self.kl_var = (
            -0.5 * 2.0 * self.sum_log_diag_L_Q0
            + 0.5 * tf.linalg.trace(self.Q0 @ self.V)
            - 0.5 * self.n_weights
        )
        self.kl_mean = +0.5 * tf.einsum(
            "n,nm,m", self.mean_diff, self.Q0, self.mean_diff
        )
        if self.kl_no_corl:
            for i in range(self.kl_n_blocks):
                idx_l = (self.n_weights // self.kl_n_blocks) * i
                idx_u = (self.n_weights // self.kl_n_blocks) * (i + 1)
                V_block = self.V[idx_l:idx_u, idx_l:idx_u]
                self.kl_var += -0.5 * tf.linalg.logdet(V_block)
        else:
            self.kl_var += -0.5 * 2.0 * tf.reduce_sum(log_diag_L_V)

        self.kl = tfp.distributions.kl_divergence(self.w_dist, self.prior)
        self.kl_old = self.kl_mean + self.kl_var

        # Sum up constraints
        self.c_sum = tf.constant(0.0, dtype=self.floatT)
        c_list = tf.zeros(self.n_t_const, dtype=self.floatT)
        for c, lagrange_var, idx in zip(self.const, lagrange_list, self.c_scatter_idx):
            c_val = c.evaluate(self.marginals, tf.ones_like(lagrange_var))
            c_list = tf.tensor_scatter_nd_update(c_list, idx, tf.reshape(c_val, [-1]))
            self.c_sum += tf.reduce_sum(c_val * lagrange_var)

        # Lagrange loss
        self.loss = self.kl * self.kl_scale + self.c_sum + self.p_sum

        return self.loss, c_list, self.kl, self.p_sum

    def set_kl_scale(self, scale):
        self.kl_scale = tf.constant(scale, dtype=self.floatT)

    def set_prior(self, m0, V0, Q0=None):
        self.m0 = m0
        self.V0 = V0
        self.L_V0 = sp.linalg.cholesky(V0, lower=True)
        if Q0 is None:
            self.Q0 = sp_spd_inv(V0)
        else:
            self.Q0 = Q0
        self.L_Q0 = sp.linalg.cholesky(self.Q0, lower=True)
        self.sum_log_diag_L_Q0 = np.sum(np.log(np.diagonal(self.L_Q0)))
        self.prior = tfp.distributions.MultivariateNormalFullCovariance(m0, V0)

    def get_initial(self, assign=True):
        if assign:
            # Mean
            M0 = np.reshape(self.m0, [self.n_obs, self.n_basis]).transpose()

            # V diagonal
            L_V0_log_diag = np.log(np.diagonal(self.L_V0))

            # V off-diagonal
            L_V0_off_diag = tf.cast(
                tfp.math.fill_triangular_inverse(self.L_V0), self.floatT
            )
            return np.concatenate(
                [M0.flatten(), L_V0_log_diag.flatten(), L_V0_off_diag.numpy().flatten()]
            )
        else:
            return np.zeros(
                int(
                    self.n_weights
                    + self.n_weights
                    + (self.n_weights) * (self.n_weights + 1) / 2
                )
            )

    def update_lagrange(self):
        self.lagrange_updates += 1

    def _unroll_var(self):
        def f(x):
            start = 0
            end = self.n_basis * self.n_obs
            m = x[start:end]
            M = tf.reshape(m, [self.n_basis, self.n_obs])
            start = end
            end += self.n_basis * self.n_obs
            log_diag_L_V = x[start:end]
            start = end
            off_diag_L_V = x[start:]
            return M, log_diag_L_V, off_diag_L_V

        return f

    def _closure(self, x):
        return tfp.math.value_and_gradient(
            lambda x: self.tf_fun_graph(*self._unroll_var()(x)), x
        )

    def _callback(self, solution):
        paths = self.sample(self.n_paths_violations, x_opt=solution.position)
        # self.var_hist.append(solution.position.numpy())
        self.loss_hist.append(solution.objective_value.numpy())
        self.kl_hist.append(self.kl.numpy())
        lam_vec = np.array([])
        cval_vec = np.array([])
        violations = tf.zeros([self.n_paths_violations], dtype=tf.bool)
        for c in self.const:
            violations = tf.math.logical_or(violations, c.get_violations(paths))
            lam_vec = np.concatenate(
                [lam_vec, np.reshape(c.lagrange_var.numpy(), [-1])], axis=0
            )
            cval_vec = np.concatenate(
                [cval_vec, np.reshape(c.constraint.numpy(), [-1])], axis=0
            )
        perc_violations = np.sum(violations) / self.n_paths_violations
        self.violation_hist.append(perc_violations)
        pval_vec = []
        for p in self.penalty:
            pval_vec.append(p.penalty)
        self.lam_hist.append(np.squeeze(np.array(lam_vec)))
        self.const_hist.append(np.squeeze(np.array(cval_vec)))
        self.penalty_hist.append(np.squeeze(np.array(pval_vec)))

    def _lbfgs_callback(self, state, pbar):
        c_max = tf.reduce_max(self.c_list).numpy()
        c_sum = tf.reduce_sum(self.c_list).numpy()
        pbar.set_postfix(
            {
                "fval": state.objective_value.numpy(),
                "kl": self.kl.numpy(),
                "c_sum": c_sum,
                "p_sum": self.p_sum_tmp.numpy(),
                "c_max": c_max,
                "grad_norm": tf.norm(state.objective_gradient).numpy(),
            },
            refresh=False,
        )

        c_step = c_max - self.c_max
        self.c_steps_sum += c_step - self.c_steps[0]
        # if self.c_steps_sum > 0 and c_max > 0:  # and c_sum > 0:
        #     state = state._replace(failed=tf.constant(True))
        self._push_c_steps_queue(c_step)
        self.c_max = c_max
        return state

    def _push_c_steps_queue(self, c_step):
        self.c_steps = np.concatenate((self.c_steps[1:], [c_step]), axis=0)

    def fit(
        self,
        x0,
        max_iters=int(1e4),
        sub_iters=int(1e2),
        tolerance=1e-6,
        x_tolerance=0,
        f_relative_tolerance=0,
        const_tolerance=1e-6,
        f_improved_tolerance=1e-6,
        parallel_iterations=1,
        num_correction_pairs=10,
        first_iter=0,
        n_c_steps=40,
        callback=None,
        callback_freq=5,
        n_paths_violations=int(1e5),
    ):
        """
        Parameters:
            x0: optimization starting point, use internal `get_initial` function
            max_iters: Maximum iterations of the cpmp algorithm (EMM steps)
            sub_iters: Maximum iterations of each inner-loop L-BFGS algorithm
            tolerance: L-BFGS tolerance
            x_tolerance: L-BFGS x_tolerance
            f_relative_tolerance: L-BFGS f_relative_tolerance
            const_tolerance: allowed constraint sum for the algorithm to terminate prematurely
            f_improved_tolerance: minimum improvement per lagrange update, otherwise algorithm can be terminated early
            parallel_iterations: L-BFGS parallel_iterations
            num_correction_pairs: L-BFGS num_correction_pairs
            first_iter: Maximum iterations for the first L-BFGS call
            n_c_steps: number of constraint steps to be considered for early abort criterion of the inner loop (currently disabled)
            callback: custom callback, gets called after every `callback_freq` inner loop iteration and gets the L-BFGS output
            callback_freq: callback frequency
            n_paths_violations: Paths sampled for evaluating constraint violations
        """
        self.var_hist = []
        self.loss_hist = []
        self.kl_hist = []
        self.lam_hist = []
        self.const_hist = []
        self.penalty_hist = []
        self.violation_hist = []

        self.n_paths_violations = n_paths_violations
        self.opt_iters = 0
        self.opt_evals = 0
        self.lam_iters = 0
        unroll = self._unroll_var()

        self.n_t_const = 0
        self.c_scatter_idx = []
        for c in self.const:
            c_size = tf.size(c.lagrange_var)
            self.c_scatter_idx.append(tf.range(c_size)[:, tf.newaxis] + self.n_t_const)
            self.n_t_const += c_size.numpy()
        self.c_list = tf.zeros(self.n_t_const, dtype=self.floatT)

        opt_res = tfp.optimizer.lbfgs_minimize(
            self._closure, x0, max_iterations=first_iter
        )
        _ = self.graph(*unroll(opt_res.position))
        self._callback(opt_res)
        if not callback is None:
            callback(opt_res)
        pbar = tqdm(total=max_iters, desc="CProMP")
        while True:
            self.c_max = 1e6
            self.c_steps = -np.ones(n_c_steps) * 1e6
            self.c_steps_sum = np.sum(self.c_steps)

            opt_res = tfp.optimizer.lbfgs_minimize(
                self._closure,
                opt_res.position,
                max_iterations=sub_iters,
                tolerance=tolerance,
                x_tolerance=x_tolerance,
                f_relative_tolerance=f_relative_tolerance,
                parallel_iterations=parallel_iterations,
                num_correction_pairs=num_correction_pairs,
                one_step_callback=self._lbfgs_callback,
                # initial_inverse_hessian_estimate=opt_res.inverse_hessian_estimate,
            )

            _ = self.graph(*unroll(opt_res.position))
            self._callback(opt_res)

            max_lam = 0.0
            for c in self.const:
                c.update(opt_res.position)
                max_lam = np.maximum(max_lam, c.lagrange_var.numpy().max())
            self.lam_iters += 1
            new_loss, new_grad = self._closure(opt_res.position)

            pbar.set_postfix(
                {
                    "fval": opt_res.objective_value.numpy(),
                    "kl": self.kl.numpy(),
                    "c_sum": self.c_sum.numpy(),
                    "p_sum": self.p_sum.numpy(),
                    "max_lam": max_lam,
                    "violations": self.violation_hist[-1],
                },
                refresh=False,
            )
            pbar.update(n=1)  # may trigger a refresh

            self.opt_iters += opt_res.num_iterations.numpy()
            self.opt_evals += opt_res.num_objective_evaluations.numpy()

            # Callback
            if self.lam_iters % callback_freq == 0:
                if not (callback is None):
                    callback(opt_res)

            if self.lam_iters > max_iters:
                break
            print(f"f_improved: {(new_loss - opt_res.objective_value).numpy()}")
            if (
                opt_res.converged
                & ((new_loss - opt_res.objective_value) < f_improved_tolerance)
                & (self.c_sum < const_tolerance)
            ):
                break
        pbar.close()
        return opt_res

    def sample(self, number_of_samples, x_opt=None):
        if not x_opt is None:
            unroll = self._unroll_var()
            _ = self.graph(*unroll(x_opt))
        w_samples = self.w_dist.sample(number_of_samples).numpy()
        w_samples = np.transpose(
            w_samples.reshape([number_of_samples, self.n_obs, self.n_basis]), [0, 2, 1]
        )
        y_samples = np.einsum("tb,sbo->sto", self.X, w_samples)
        return y_samples

    def set_kl_computation(self, no_corl, n_blocks):
        self.kl_no_corl = no_corl
        self.kl_n_blocks = n_blocks


def main():
    import pickle

    import matplotlib.pyplot as plt

    from opt_pmp_utils import tf_allow_growth

    fprimitive = "real_robot/weights_grasp_lowVar_2020-01-29_16:00:06_processed"
    assign = False

    with open(fprimitive, mode="rb") as f:
        primitive = pickle.load(f)

    const = []
    np.random.seed(7)
    n_b = 20
    n_b = primitive.n_basis
    n_o = 7
    n_o = primitive.n_outputs

    n_w = n_b * n_o
    m0 = np.random.randn(n_w)
    V0 = np.random.randn(n_w, n_w)
    V0 = 0.5 * (V0 + V0.T)
    V0 = V0 + n_w * np.eye(n_w)
    m0 = primitive.m_w
    V0 = primitive.V_w

    bf = GaussianRBF(
        n_b,
        np.linspace(0, 1, primitive.n_ts),
        std_distance=1.0,
        normalize_features=True,
        c_t_delta=0.1,
    )

    model = CProMP(n_o, basis_fun=bf, const=const, floatT=tf.float64)
    model.set_prior(m0, V0, Q0=None)

    x0 = tf.constant(model.get_initial(assign))

    opt_res = model.fit(x0, max_iters=1000)

    print("Converged: {}".format(opt_res.converged))
    print("iterations: {}".format(model.opt_iters))
    print("lagrange_updates: {}".format(model.lam_iters))
    print("evaluations: {}".format(model.opt_evals))
    x_opt = opt_res.position
    M, log_diag_L_V, off_diag_L_V = model._unroll_var()(x_opt)
    model.graph(M, log_diag_L_V, off_diag_L_V)
    m_opt = model.M_vec
    V_opt = model.V

    fig, axs = plt.subplots(4, 1)
    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=3)

    ax1.plot(m0, label="prior")
    ax1.plot(m_opt, label="posterior")
    ax1.set_ylabel("Mean")
    ax1.legend()
    image = ax2.imshow(np.abs(V_opt - V0))
    ax2.set_title("V - V0")
    plt.colorbar(image, ax=ax2)
    plt.show()


if __name__ == "__main__":
    main()
