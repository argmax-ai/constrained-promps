import numpy as np
import tensorflow_probability as tfp

from opt_pmp_utils.basis_functions import GaussianRBF
from opt_pmp_utils.promp import ProMP


class TrajectoryEnv(object):
    """Creates a 2D-Test environment with a prior ProMP and a range
    of potential constraints, specifically:
        - obstacles for repeller constraints,
        - via-points for waypoint constraints,
        - virtual walls

    """

    def __init__(self, n_dim, nt, nbasis_pmp, **kwargs):
        super(TrajectoryEnv, self).__init__()
        self.n_dim = n_dim
        self.nt = nt
        self.nbasis_pmp = nbasis_pmp
        self.t = np.linspace(0, 1, self.nt)

        # Gather default arguments
        kwargs, bfun_args, pmp_args = self._defaults(**kwargs)
        self.kwargs = kwargs

        # Setup basis function and ProMP
        self.bfun = GaussianRBF(n_basis=self.nbasis_pmp, time_vec=self.t, **bfun_args)
        self.pmp = ProMP(number_of_outputs=n_dim, basis_fun=self.bfun, **pmp_args)
        self.pmp.m0[: self.pmp.n_basis] = np.linspace(
            kwargs["x_min"], kwargs["x_max"], self.pmp.n_basis
        )

        # Sample via-points
        self.via_points = []
        x_points = np.linspace(kwargs["x_min"], kwargs["x_max"], kwargs["n_via_points"])
        delta = kwargs["x_max"] - kwargs["x_min"]
        # x_points[1] = np.random.uniform(0.25 * delta, 0.75 * delta) + kwargs["x_min"]
        y_points = np.random.uniform(
            kwargs["y_min"], kwargs["y_max"], kwargs["n_via_points"]
        )
        idx_points = np.linspace(0, self.nt - 1, kwargs["n_via_points"])
        # idx_points[1] = np.random.uniform(0.33 * nt, 0.66 * nt)
        idx_points = idx_points.astype(np.int)
        for x, y, idx in zip(x_points, y_points, idx_points):
            self.via_points.append({"point": [x, y], "idx": idx})
            self.pmp.add_via_point(
                [x, y],
                np.diag(np.ones(n_dim) * kwargs["via_point_var"]),
                self.pmp.X[idx],
            )

        # Project ProMP
        self.pmp.project()
        self.w_dist = tfp.distributions.MultivariateNormalFullCovariance(
            self.pmp.m0, self.pmp.V0
        )
        self.pmp_dist = tfp.distributions.MultivariateNormalFullCovariance(
            self.pmp.myt, self.pmp.Vyt + np.diag(np.ones(nt * n_dim) * 1e-6)
        )
        self.marginals = tfp.distributions.MultivariateNormalFullCovariance(
            self.pmp.my_pp, self.pmp.Vy_pp
        )

        # Sample obstacles
        self.obstacles = []
        x_points = np.random.uniform(
            kwargs["obstacle_delta_x_min"],
            kwargs["obstacle_delta_x_max"],
            kwargs["n_obstacles"],
        )
        y_points = np.random.uniform(
            kwargs["obstacle_delta_y_min"],
            kwargs["obstacle_delta_y_max"],
            kwargs["n_obstacles"],
        )
        delta = 1 - kwargs["obstacles_field"]
        idx = np.round(
            np.random.uniform(delta * nt, (1 - delta) * nt, kwargs["n_obstacles"])
        )
        # margins = np.tile(
        #     np.random.uniform(kwargs["rep_margin_min"], kwargs["rep_margin_max"]),
        #     kwargs["n_obstacles"],
        # )
        margins = np.random.uniform(
            kwargs["rep_margin_min"],
            kwargs["rep_margin_max"],
            kwargs["n_obstacles"],
        )
        for x, y, margin, i in zip(x_points, y_points, margins, idx):
            self.obstacles.append(
                {
                    "point": self.marginals[i].mean().numpy() + [x, y],
                    "margin": margin,
                    "delta": [x, y],
                }
            )

        # Sample temp unbound via-points
        self.t_via_points = []
        x_points = (
            np.random.uniform(0.4, 0.6, kwargs["n_t_via_points"])
            * (kwargs["x_max"] - kwargs["x_min"])
            / (kwargs["n_t_via_points"] + 2)
            + np.linspace(
                kwargs["x_min"], kwargs["x_max"], kwargs["n_t_via_points"] + 3
            )[1:-2]
        )
        y_mean = []
        for x in x_points:
            idx = np.argmin(np.abs(self.marginals.mean()[:, 0] - x))
            y_mean.append(self.marginals.mean()[idx, 1].numpy())
        y_points = (
            np.random.uniform(
                kwargs["t_via_y_min"], kwargs["t_via_y_max"], kwargs["n_t_via_points"]
            )
            + y_mean
        )
        margins = np.random.uniform(
            kwargs["tVia_margin_min"],
            kwargs["tVia_margin_max"],
            kwargs["n_t_via_points"],
        )
        for x, y, margin in zip(x_points, y_points, margins):
            self.t_via_points.append(
                {
                    "point": [x, y],
                    "margin": margin,
                }
            )

        # Sample virtual walls
        self.v_walls = []
        start = np.array(self.via_points[0]["point"])
        end = np.array(self.via_points[-1]["point"])
        mid = (start + end) / 2
        delta_vec = end - start
        while len(self.v_walls) < kwargs["n_v_walls"]:
            if np.random.uniform() < 0.5:
                d_vec = self._rotate2dVector(delta_vec, np.pi / 2)
            else:
                d_vec = self._rotate2dVector(delta_vec, -np.pi / 2)
            d_vec = (
                d_vec
                / np.linalg.norm(d_vec)
                * np.random.uniform(kwargs["v_wall_d_min"], kwargs["v_wall_d_max"])
            )
            b = mid + d_vec
            rot = np.random.uniform(-1.0, 1.0) * 2 * np.pi - np.pi
            n_vec = self._rotate2dVector(d_vec, rot)
            if np.dot(end - b, n_vec) > 0 or np.dot(start - b, n_vec) > 0:
                continue
            self.v_walls.append({"b": b, "n_vec": n_vec})

    def _rotate2dVector(cls, vec, rotation):
        rotMat = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        return np.einsum("ij,...j->...i", rotMat, vec)

    def sample(self, n_samples):
        return self.pmp.sample(n_samples)

    def _defaults(self, **kwargs):
        """
        Default arguments, split in three categories:
            - Basis function arguments (bfun_args), concerning the
                ProMP basis functions
            - ProMP arguments (pmp_args), used during the creation
                of the ProMP
            - Environment arguments (kwargs), concerning the trajectory
                environment itself. Examples are the number of
                obstacles/ via-points, or the margins they require

        All arguments can be overwritten through kwargs given to the
        Environment creation
        """
        bfun_args = {}
        pmp_args = {}

        if not "c_t_delta" in kwargs:
            bfun_args["c_t_delta"] = 0.1
        else:
            bfun_args["c_t_delta"] = kwargs["c_t_delta"]

        if not "normalize_features" in kwargs:
            bfun_args["normalize_features"] = True
        else:
            bfun_args["normalize_features"] = kwargs["normalize_features"]

        if not "std_distance" in kwargs:
            bfun_args["std_distance"] = 1.0
        else:
            bfun_args["std_distance"] = kwargs["std_distance"]

        if not "pmp_prior_var" in kwargs:
            pmp_args["std_prior"] = 1.0
        else:
            pmp_args["std_prior"] = kwargs["pmp_prior_var"]
        if not "x_min" in kwargs:
            kwargs["x_min"] = -3.0
        if not "x_max" in kwargs:
            kwargs["x_max"] = 3.0
        if not "y_min" in kwargs:
            kwargs["y_min"] = -3.0
        if not "y_max" in kwargs:
            kwargs["y_max"] = 3.0

        # Obstacle arguments
        if not "obstacle_delta_x_min" in kwargs:
            kwargs["obstacle_delta_x_min"] = -0.0
        if not "obstacle_delta_x_max" in kwargs:
            kwargs["obstacle_delta_x_max"] = 0.0
        if not "obstacle_delta_y_min" in kwargs:
            kwargs["obstacle_delta_y_min"] = -1.5
        if not "obstacle_delta_y_max" in kwargs:
            kwargs["obstacle_delta_y_max"] = 1.5

        # Initial ProMP via-point arguments
        if not "n_via_points" in kwargs:
            kwargs["n_via_points"] = 3
        if not "n_obstacles" in kwargs:
            kwargs["n_obstacles"] = 0
        if not "via_point_var" in kwargs:
            kwargs["via_point_var"] = 1e-1
        if not "rep_margin_min" in kwargs:
            kwargs["rep_margin_min"] = 0.5
        if not "rep_margin_max" in kwargs:
            kwargs["rep_margin_max"] = 1.6
        if not "obstacles_field" in kwargs:
            kwargs["obstacles_field"] = 0.8

        # Arguments for the constraint via-points
        if not "n_t_via_points" in kwargs:
            kwargs["n_t_via_points"] = 0
        if not "t_via_y_min" in kwargs:
            kwargs["t_via_y_min"] = -1.5
        if not "t_via_y_max" in kwargs:
            kwargs["t_via_y_max"] = 1.5
        if not "tVia_margin_min" in kwargs:
            kwargs["tVia_margin_min"] = 0.05
        if not "tVia_margin_max" in kwargs:
            kwargs["tVia_margin_max"] = 0.2

        # Virtual wall arguments
        if not "n_v_walls" in kwargs:
            kwargs["n_v_walls"] = 0
        if not "v_wall_d_min" in kwargs:
            kwargs["v_wall_d_min"] = 0.2
        if not "v_wall_d_max" in kwargs:
            kwargs["v_wall_d_max"] = 1.5

        return kwargs, bfun_args, pmp_args

    def _show_marginals(self, ax, dist, idx, color="b", alpha=0.2):
        ax.plot(self.t, dist.mean()[:, idx], c=color)
        ax.fill_between(
            self.t,
            dist.mean()[:, idx] - 3 * dist.stddev()[:, 0],
            dist.mean()[:, idx] + 3 * dist.stddev()[:, 0],
            color=color,
            alpha=alpha,
        )
        ax.set_xlabel("Time")

    def _show_2d(self, ax, show_obs):
        from matplotlib.patches import Circle

        if show_obs:
            for ob in self.obstacles:
                ax.scatter(
                    ob["point"][0], ob["point"][1], c="k", marker="+", s=20, zorder=4
                )
                ax.add_artist(
                    Circle(
                        ob["point"], radius=ob["margin"], ec="k", fc="None", zorder=4
                    )
                )

            for viaP in self.t_via_points:
                ax.scatter(
                    viaP["point"][0],
                    viaP["point"][1],
                    c="k",
                    marker="+",
                    s=20,
                    zorder=4,
                )
                ax.add_artist(
                    Circle(
                        viaP["point"],
                        radius=viaP["margin"],
                        ec="k",
                        fc="None",
                        zorder=4,
                    )
                )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim([self.kwargs["x_min"] - 1.0, self.kwargs["x_max"] + 1.0])
        ax.set_ylim([self.kwargs["y_min"] - 1.0, self.kwargs["y_max"] + 1.0])
        ax.set_aspect("equal", "box")

    def _show_2d_dist(
        self, ax, dist, marginals, n_samples, color="b", cov_alpha=0.3, path_alpha=0.3
    ):
        from opt_pmp_utils.plot_2d_normal import plot2dNormal

        paths = dist.sample(n_samples)
        ax.scatter(
            marginals.mean()[:, 0],
            marginals.mean()[:, 1],
            c=color,
            marker=".",
            s=1,
            zorder=3,
        )
        for path in paths:
            if path.shape[-1] == self.n_dim:
                ax.plot(path[:, 0], path[:, 1], "k", alpha=path_alpha, zorder=5)
            else:
                ax.plot(
                    path[: self.nt], path[self.nt :], "k", alpha=path_alpha, zorder=5
                )

        for i in range(marginals.batch_shape_tensor()[0]):
            plot2dNormal(
                marginals.mean()[i],
                marginals.covariance()[i],
                ax,
                color="None",
                fc=color,
                alpha=cov_alpha,
            )

    def show(self, n_samples, show_obs=True):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(4, 2)
        ax = []
        ax.append(plt.subplot2grid((4, 2), (0, 0), colspan=1))
        ax.append(plt.subplot2grid((4, 2), (0, 1), colspan=1))
        ax.append(plt.subplot2grid((4, 2), (1, 0), colspan=2, rowspan=3))
        self._show_marginals(ax[0], self.marginals, 0, color="b", alpha=0.2)
        ax[0].set_ylabel("x")
        self._show_marginals(ax[1], self.marginals, 1, color="b", alpha=0.2)
        ax[1].set_ylabel("y")
        self._show_2d(ax[2], show_obs)
        self._show_2d_dist(
            ax[2],
            self.pmp_dist,
            self.marginals,
            n_samples,
            color="b",
            cov_alpha=0.3,
            path_alpha=0.3,
        )
        fig.tight_layout()
        return fig, ax

    def compare_show(self, n_samples, opt_dist, marginals, show_obs=True):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(4, 2)
        ax = []
        ax.append(plt.subplot2grid((4, 2), (0, 0), colspan=1))
        ax.append(plt.subplot2grid((4, 2), (0, 1), colspan=1))
        ax.append(plt.subplot2grid((4, 2), (1, 0), colspan=1, rowspan=3))
        ax.append(plt.subplot2grid((4, 2), (1, 1), colspan=1, rowspan=3))
        self._show_marginals(ax[0], self.marginals, 0, color="b", alpha=0.2)
        self._show_marginals(ax[0], marginals, 0, color="r", alpha=0.2)
        ax[0].set_ylabel("x")
        self._show_marginals(ax[1], self.marginals, 1, color="b", alpha=0.2)
        self._show_marginals(ax[1], marginals, 1, color="r", alpha=0.2)
        ax[1].set_ylabel("y")
        self._show_2d(ax[2], show_obs)
        self._show_2d_dist(
            ax[2],
            self.pmp_dist,
            self.marginals,
            n_samples,
            color="b",
            cov_alpha=0.3,
            path_alpha=0.3,
        )
        self._show_2d(ax[3], show_obs)
        self._show_2d_dist(
            ax[3],
            opt_dist,
            marginals,
            n_samples,
            color="r",
            cov_alpha=0.3,
            path_alpha=0.3,
        )
        ax[0].set_title("x-Marginal")
        ax[1].set_title("y-Marginal")
        ax[2].set_title("Original")
        ax[3].set_title("Adapted")
        fig.tight_layout()
        return fig, ax


def main():
    import matplotlib.pyplot as plt

    env = TrajectoryEnv(
        2, 21, 8, c_t_delta=0.1, pmp_prior_var=2.0, n_obstacles=0, n_t_via_points=3
    )
    env.show(n_samples=10, show_obs=True)
    plt.show()


if __name__ == "__main__":
    main()
