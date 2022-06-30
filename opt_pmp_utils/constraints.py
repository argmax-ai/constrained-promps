import numpy as np
import opt_einsum
import scipy as sp
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from matplotlib.patches import FancyArrowPatch

from opt_pmp_utils.unscented_transform import uTransform


def FK(q):
    x = tf.cos(q[..., 0]) + tf.cos(q[..., 0] + q[..., 1])
    y = tf.sin(q[..., 0]) + tf.sin(q[..., 0] + q[..., 1])
    return tf.stack([x, y], axis=1)


class Arrow3d(FancyArrowPatch):
    def __init__(self, x_dx, y_dy, z_dz, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self.x_dx = x_dx
        self.y_dy = y_dy
        self.z_dz = z_dz

    def draw(self, renderer):
        from mpl_toolkits.mplot3d import proj3d

        x, y, z = proj3d.proj_transform(self.x_dx, self.y_dy, self.z_dz, renderer.M)
        self.set_positions((x[0], y[0]), (x[1], y[1]))
        FancyArrowPatch.draw(self, renderer)


class Constraint(object):
    """Base class for different constraints in the CPMP framework

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space
        floatT: tensorflow type used for variable initialisation
    """

    def __init__(
        self,
        lagrange_learning_rate: float,
        alpha: float,
        transform_fcn,
        floatT=tf.float64,
    ):
        super(Constraint, self).__init__()
        self.lagrange_learning_rate = lagrange_learning_rate
        self.alpha = alpha
        self.transform_fcn = transform_fcn
        self.floatT = floatT

    def evaluate(self):
        """Returns the constraint cost value (lag. mult. * const.)"""
        raise NotImplementedError()

    def update(self):
        """Updates the lag. mult. given the current opt. state"""
        raise NotImplementedError()

    def plot2D(self, ax):
        """Plots the constraint in a 2D-plot"""
        raise NotImplementedError()

    def plot3D(self, ax):
        """Plots the constraint in a 3D-plot"""
        raise NotImplementedError()

    def get_violations(self, paths):
        """Returns boolean mask of which paths violate the constraint"""
        return tf.zeros(paths.shape[0], dtype=tf.bool)
        raise NotImplementedError()

    def info(self):
        """Prints information about the constraint"""
        print("\tAlpha:\t\t{}\n".format(self.alpha))
        print("\tTransformation:\t\t{}\n".format(self.transform_fcn))
        print("\tTime Mask:\t\t{}\n".format(self.time_mask))


class SquareDistance2Point(Constraint):
    """Base class for distance based constraints

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space
        point: obstacle/waypoint location
        margin: spatial margin around obstacle/waypoint
        n_timesteps: number points in the timegrid
        time_mask: indicated at which timepoints the constraint is active (=1)
        floatT: tensorflow type used for variable initialisation
        lagrange_initial: initial value for lagrange multiplier
    """

    def __init__(
        self,
        lagrange_learning_rate,
        alpha,
        transform_fcn,
        point,
        margin,
        n_timesteps,
        time_mask=1,
        floatT=tf.float64,
        lagrange_initial=1.0,
    ):
        with tf.variable_scope("SquareDistance2Point"):
            super(SquareDistance2Point, self).__init__(
                lagrange_learning_rate,
                alpha,
                transform_fcn,
                floatT=floatT,
            )
            self.lagrange_var = tf.squeeze(
                tf.Variable(
                    initial_value=lagrange_initial * np.ones(n_timesteps),
                    dtype=self.floatT,
                )
            )
        self.lagrange_initial = lagrange_initial
        self.point = point
        self.margin = margin
        self.time_mask = time_mask
        self.n_timesteps = n_timesteps
        self._color = "k"
        self._init()

    def _init(self):
        pass

    def transform(self, q):
        d = self.transform_fcn(q) - self.point
        sqd = tf.einsum("...n,...n->...", d, d)
        return tf.expand_dims(sqd, -1)

    def evaluate(self, dist, lagrange_var, alpha=1e-1, kappa=0, beta=2.0):
        self.mean, self.covar = uTransform(
            dist, self.transform, alpha=alpha, kappa=kappa, beta=beta
        )
        self.mean = tf.squeeze(self.mean)
        self.covar = tf.squeeze(self.covar)
        self.distd = self._target_dist()
        self.mass = self.distd.cdf((self.margin) ** 2)
        self.constraint = tf.squeeze(self._compute_constraint() * self.time_mask)
        return self.constraint * lagrange_var

    def evaluate_internal(self, dist):
        return tf.reduce_sum(self.evaluate(dist, self.lagrange_var))

    def update(self, x):
        # print("Lagrange update")
        self.lagrange_var = self.lagrange_var * tf.exp(
            self.lagrange_learning_rate * self.constraint
        )

    def plot2D(self, ax):
        from matplotlib.patches import Circle

        ax.scatter(self.point[0], self.point[1], c=self._color, marker="+", s=20)
        ax.add_artist(
            Circle(self.point, radius=self.margin, ec=self._color, fc="None", zorder=4)
        )

    def plot3D(self, ax):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.point[0] + self.margin * np.outer(np.cos(u), np.sin(v))
        y = self.point[1] + self.margin * np.outer(np.sin(u), np.sin(v))
        z = self.point[2] + self.margin * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(
            x,
            y,
            z,
            color=self._color,
            ec="k",
            linewidth=0.3,
            alpha=0.4,
            rstride=10,
            cstride=10,
        )
        ax.scatter(
            self.point[0], self.point[1], self.point[2], c=self._color, marker="X", s=50
        )

    def _compute_constraint(self):
        raise NotImplementedError()

    def _target_dist(self):
        raise NotImplementedError()

    def info(self):
        print("\tPoint:\t\t{}\n".format(self.point))
        print("\tMargin:\t\t{}\n".format(self.margin))
        if hasattr(self, "lagrange_initial"):
            print("\tLagrange initial:\t\t{}\n".format(self.lagrange_initial))
        print("\tLagrange final:\t\t{}\n".format(self.lagrange_var.numpy))
        super(SquareDistance2Point, self).info()


class Repeller(SquareDistance2Point):
    """Repeller constraint

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space
        x_avoid: obstacle location
        margin: spatial margin around obstacle
        n_timesteps: number points in the timegrid
        time_mask: indicated at which timepoints the constraint is active (=1)
        floatT: tensorflow type used for variable initialisation
        lagrange_initial: initial value for lagrange multiplier
    """

    def __init__(
        self,
        lagrange_learning_rate,
        alpha,
        transform_fcn,
        x_avoid,
        margin,
        n_timesteps,
        time_mask=1,
        floatT=tf.float64,
        lagrange_initial=1.0,
    ):
        with tf.variable_scope("Repeller"):
            super(Repeller, self).__init__(
                lagrange_learning_rate,
                alpha,
                transform_fcn,
                x_avoid,
                margin,
                n_timesteps,
                time_mask=time_mask,
                floatT=floatT,
                lagrange_initial=lagrange_initial,
            )
        self._color = "k"

    def _compute_constraint(self):
        return self.mass - self.alpha

    def _target_dist(self):
        self.rate = self.mean / tf.squeeze(self.covar)
        self.concentration = self.mean**2 / tf.squeeze(self.covar)
        return tfp.distributions.Gamma(self.concentration, self.rate)

    def get_violations(self, paths):
        # paths: sxtxd
        distances = tf.norm(self.point - paths, axis=-1)
        min_distance = tf.reduce_min(distances, axis=-1)
        violations = min_distance < self.margin
        return violations

    def info(self):
        print("Repeller constraint:\n")
        super(Repeller, self).info()
        print("\n")


class Waypoint(SquareDistance2Point):
    """Waypoint constraint

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space
        waypoint: waypoint location
        margin: allowed spatial margin around waypoint
        n_timesteps: number points in the timegrid
        time_mask: indicated at which timepoints the constraint is active (=1)
        floatT: tensorflow type used for variable initialisation
        lagrange_initial: initial value for lagrange multiplier
    """

    def __init__(
        self,
        lagrange_learning_rate,
        alpha,
        transform_fcn,
        waypoint,
        margin,
        n_timesteps,
        time_mask=1,
        floatT=tf.float64,
        lagrange_initial=1.0,
    ):
        with tf.variable_scope("Waypoint"):
            super(Waypoint, self).__init__(
                lagrange_learning_rate,
                alpha,
                transform_fcn,
                waypoint,
                margin,
                n_timesteps,
                time_mask=time_mask,
                floatT=tf.float64,
                lagrange_initial=lagrange_initial,
            )

    def _compute_constraint(self):
        return 1 - self.mass - self.alpha

    def _target_dist(self):
        self.rate = self.mean / tf.squeeze(self.covar)
        self.concentration = self.mean**2 / tf.squeeze(self.covar)
        return tfp.distributions.Gamma(self.concentration, self.rate)

    def get_violations(self, paths):
        # paths: sxtxd
        distances = tf.norm(self.point - paths, axis=-1)
        min_distance = tf.reduce_min(distances, axis=-1)
        violations = min_distance > self.margin
        return violations

    def info(self):
        print("Waypoint constraint:\n")
        super(Waypoint, self).info()
        print("\n")


class OneTimeWaypoint(Waypoint):
    """Temporally unbound waypoint constraint

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space
        waypoint: waypoint location
        margin: allowed spatial margin around waypoint
        n_timesteps: number points in the timegrid
        time_mask: indicated at which timepoints the constraint is active (=1)
        floatT: tensorflow type used for variable initialisation
        lagrange_initial: initial value for lagrange multiplier
        window_margin: constraint enforces being at waypoint for
            2 * window_margin + 1 timesteps
    """

    def __init__(
        self,
        lagrange_learning_rate,
        alpha,
        transform_fcn,
        waypoint,
        margin,
        n_timesteps,
        floatT=tf.float64,
        lagrange_initial=1.0,
        window_margin=1,
    ):
        with tf.variable_scope("Waypoint"):
            super(Waypoint, self).__init__(
                lagrange_learning_rate,
                alpha,
                transform_fcn,
                waypoint,
                margin,
                1,
                time_mask=1.0,
                floatT=tf.float64,
                lagrange_initial=lagrange_initial,
            )
        self.window_margin = window_margin
        self.window_size = 2 * window_margin + 1
        self.n_ts = n_timesteps

    def evaluate(self, dist, lagrange_var):
        self.mean, self.covar = uTransform(dist, self.transform)
        self.mean = tf.squeeze(self.mean)
        self.covar = tf.squeeze(self.covar)
        self.distd = self._target_dist()
        self.mass = self.distd.cdf((self.margin) ** 2)
        con_vals = self._compute_constraint() * self.time_mask
        self.min_idx = tf.math.argmin(con_vals)
        gather_start = tf.math.maximum(
            tf.constant(0, dtype=tf.dtypes.int64), self.min_idx - self.window_margin
        )
        gather_start = tf.math.minimum(
            tf.constant(self.n_ts - self.window_size, dtype=tf.dtypes.int64),
            gather_start,
        )
        gather_range = tf.range(self.window_size, dtype=tf.dtypes.int64) + gather_start
        self.constraint = tf.reduce_mean(tf.gather(con_vals, gather_range))
        return self.constraint * lagrange_var

    def info(self):
        print("OneTimeWaypoint constraint:\n")
        super(OneTimeWaypoint, self).info()
        print("\n")


class DualRobotAvoidance(SquareDistance2Point):
    """Dual robot avoidance constraint

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space, usually
            robot forward kinematics to the two corresponding robot links
        margin: desired distance between the robots
        n_timesteps: number points in the timegrid
        time_mask: indicated at which timepoints the constraint is active (=1)
        floatT: tensorflow type used for variable initialisation
        lagrange_initial: initial value for lagrange multiplier
    """

    def __init__(
        self,
        lagrange_learning_rate,
        alpha,
        transform_fcn,
        margin,
        n_timesteps,
        time_mask=1,
        floatT=tf.float64,
        lagrange_initial=1.0,
    ):
        with tf.variable_scope("DualRobotAvoidance"):
            super(DualRobotAvoidance, self).__init__(
                lagrange_learning_rate,
                alpha,
                transform_fcn,
                0.0,
                margin,
                n_timesteps,
                time_mask=time_mask,
                floatT=floatT,
                lagrange_initial=lagrange_initial,
            )
        self._color = "k"

    def _init(self):
        pass

    def transform(self, q):
        xA, xB = self.transform_fcn(q)
        d = xA - xB
        sqd = tf.einsum("...n,...n->...", d, d)
        return tf.expand_dims(sqd, -1)

    def _compute_constraint(self):
        return self.mass - self.alpha

    def _target_dist(self):
        self.rate = self.mean / tf.squeeze(self.covar)
        self.concentration = self.mean**2 / tf.squeeze(self.covar)
        return tfp.distributions.Gamma(self.concentration, self.rate)

    def info(self):
        print("DualRobotAvoidance constraint:\n")
        super(DualRobotAvoidance, self).info()
        print("\n")


class ConvexConstraint(Constraint):
    """Convex constraint, can be used for virtual walls

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space
        normal_vectors: normal vector of the virtual wall
        intersection: one point on the virtual wall
        n_timesteps: number points in the timegrid
        time_mask: indicated at which timepoints the constraint is active (=1)
        floatT: tensorflow type used for variable initialisation
        lagrange_initial: initial value for lagrange multiplier
        vel_decay: unused?
    """

    def __init__(
        self,
        lagrange_learning_rate,
        alpha,
        transform_fcn,
        normal_vectors,
        intersection,
        n_timesteps,
        time_mask=1,
        floatT=tf.float64,
        lagrange_initial=1.0,
        vel_decay=0.1,
    ):
        self.normal_vectors_np = np.array(normal_vectors)
        self.intersection_np = np.array(intersection)
        assert self.intersection_np.shape == self.normal_vectors_np.shape
        if len(self.normal_vectors_np.shape) == 1:
            self.normal_vectors_np = self.normal_vectors_np[np.newaxis, :]
            self.intersection_np = self.intersection_np[np.newaxis, :]
        with tf.variable_scope("ConvexConstraint"):
            super(ConvexConstraint, self).__init__(
                lagrange_learning_rate,
                alpha,
                transform_fcn,
                floatT=floatT,
            )
            self.normal_vectors = tf.constant(
                value=self.normal_vectors_np, dtype=floatT
            )  # n_con x cart. dim
            self.intersection = tf.constant(
                value=self.intersection_np, dtype=floatT
            )  # n_con x cart. dim
            self.lagrange_var = tf.squeeze(
                tf.Variable(
                    initial_value=lagrange_initial
                    * np.ones([n_timesteps, self.normal_vectors_np.shape[0]]),
                    dtype=floatT,
                )
            )
            self.lagrange_var_vel = tf.squeeze(
                tf.zeros(
                    [n_timesteps, self.normal_vectors_np.shape[0]],
                    dtype=floatT,
                )
            )
            self.std_normal = tfp.distributions.Normal(
                tf.constant(0.0, dtype=floatT), tf.constant(1.0, dtype=floatT)
            )
        self.n_timesteps = n_timesteps
        self.time_mask = time_mask
        self.vel_decay = vel_decay
        self.lagrange_initial = lagrange_initial

    def get_violations(self, paths):
        # Paths = sxtxd
        # inter/normal = nxd
        distance = paths[:, :, tf.newaxis, :] - self.intersection  # sxtxnxd
        direction = tf.einsum("...nd,nd->...n", distance, self.normal_vectors)  # sxtxn
        max_direction = tf.math.reduce_max(direction, [1, 2])
        violations = max_direction > 0
        return violations

    def evaluate(self, dist, lagrange_var):
        self.mean, self.covar = uTransform(dist, self.transform_fcn)
        self.V_proj_diag = opt_einsum.contract(
            "cd,...di,ci->...c",
            self.normal_vectors,
            self.covar,
            self.normal_vectors,
            backend="tensorflow",
        )
        self.mass = self.std_normal.cdf(
            tf.einsum(
                "cd,...cd->...c",
                self.normal_vectors,
                self.mean[..., tf.newaxis, :] - self.intersection,
            )
            / tf.sqrt(self.V_proj_diag)
        )
        self.constraint = tf.squeeze((self.mass - self.alpha) * self.time_mask)

        return self.constraint * lagrange_var

    def evaluate_internal(self, dist):
        return tf.reduce_sum(self.evaluate(dist, self.lagrange_var))

    def update(self, x):
        self.lagrange_var = self.lagrange_var * tf.exp(
            self.lagrange_learning_rate * self.constraint
        )

    def _rotate2dVector(self, vec, rotation):
        rotMat = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        return np.einsum("ij,...j->...i", rotMat, vec)

    def info(self):
        print("ConvexConstraint constraint:\n")
        print("\tNormal vec:\t\t{}\n".format(self.normal_vectors_np))
        print("\tIntersection:\t\t{}\n".format(self.intersection_np))
        if hasattr(self, "lagrange_initial"):
            print("\tLagrange initial:\t\t{}\n".format(self.lagrange_initial))
        print("\tLagrange final:\t\t{}\n".format(self.lagrange_var.numpy))
        super(ConvexConstraint, self).info()
        print("\n")

    def plot2D(self, ax):
        lvecs = self._rotate2dVector(self.normal_vectors_np, np.pi / 2)
        for nvec, inter, lvec in zip(
            self.normal_vectors_np, self.intersection_np, lvecs
        ):
            linep1 = inter + 10 * lvec
            linep2 = inter - 10 * lvec
            linepx = [linep1[0], linep2[0]]
            linepy = [linep1[1], linep2[1]]
            ax.plot(linepx, linepy, c="k")
            ax.arrow(inter[0], inter[1], nvec[0], nvec[1], color="k", width=0.02)

    def plot3D(
        self, ax, n_mesh=100, color="r", ec="None", alpha=0.4, rstride=10, cstride=10
    ):
        limits = np.array([[*ax.get_xlim()], [*ax.get_ylim()], [*ax.get_zlim()]])
        pos = np.linspace(limits[:, 0], limits[:, 1], num=n_mesh)
        for nvec, inter in zip(self.normal_vectors_np, self.intersection_np):
            cond = nvec != 0
            idx = cond.nonzero()[0][0]
            cond[...] = False
            cond[idx] = True
            mesh = np.zeros([3, n_mesh, n_mesh])
            a, b = np.meshgrid(pos[:, ~cond][:, 0], pos[:, ~cond][:, 1])
            mesh[~cond, ...] = np.stack((a, b))
            mesh[cond, ...] = (
                -1
                / nvec[cond]
                * np.einsum("n,n...->...", nvec[~cond], mesh[~cond, ...])
            )
            x = inter[0] + mesh[0, ...]
            y = inter[1] + mesh[1, ...]
            z = inter[2] + mesh[2, ...]
            ax.plot_surface(
                x,
                y,
                z,
                color=color,
                ec=ec,
                alpha=alpha,
                rstride=rstride,
                cstride=cstride,
            )
            ax.add_artist(
                Arrow3d(
                    [inter[0], nvec[0]],
                    [inter[1], nvec[1]],
                    [inter[2], nvec[2]],
                    arrowstyle="-|>",
                    color=color,
                    mutation_scale=10,
                    lw=1,
                )
            )


class Border1DConstraint(Constraint):
    """1 dimensional constraint, can be used for joint limits

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space
        dir_vector: +1 = upper limit, -1 = lower limit
        border: location of the 1-D limit
        n_timesteps: number points in the timegrid
        time_mask: indicated at which timepoints the constraint is active (=1)
        floatT: tensorflow type used for variable initialisation
        lagrange_initial: initial value for lagrange multiplier
    """

    def __init__(
        self,
        lagrange_learning_rate,
        alpha,
        transform_fcn,
        dir_vector,
        border,
        n_timesteps,
        time_mask=1,
        floatT=tf.float64,
        lagrange_initial=1.0,
    ):
        self.lagrange_initial = lagrange_initial
        self.dir_vector_np = np.array(dir_vector)
        self.border_np = np.array(border)
        self.proj_dim = self.dir_vector_np.size
        if len(self.border_np.shape) == 0:
            self.border_np = self.border_np[np.newaxis, np.newaxis]
            self.border_np = np.tile(self.border_np, [n_timesteps, self.proj_dim])
        elif len(self.border_np.shape) == 1:
            if self.border_np.size == n_timesteps:
                self.border_np = self.border_np[..., np.newaxis]
                self.border_np = np.tile(self.border_np, [1, self.proj_dim])
            elif self.border_np.size == self.proj_dim:
                self.border_np = self.border_np[np.newaxis, ...]
                self.border_np = np.tile(self.border_np, [n_timesteps, 1])
            else:
                raise RuntimeError(
                    "Border vector shapes cannot be broadcasted to [n_timesteps x proj_dim]"
                )
        assert self.border_np.shape[0] == n_timesteps
        assert self.border_np.shape[1] == self.proj_dim
        with tf.variable_scope("Border1DConstraint"):
            super(Border1DConstraint, self).__init__(
                lagrange_learning_rate,
                alpha,
                transform_fcn,
                floatT=floatT,
            )
            self.dir_vector = tf.constant(
                value=self.dir_vector_np, dtype=floatT
            )  # proj. dim
            self.border = tf.constant(
                value=self.border_np, dtype=floatT
            )  # n_t x proj. dim
            self.lagrange_var = tf.constant(
                lagrange_initial * np.ones([n_timesteps, self.proj_dim]),
                dtype=floatT,
            )  # n_t x proj. dim
            self.std_normal = tfp.distributions.Normal(
                tf.constant(0.0, dtype=floatT), tf.constant(1.0, dtype=floatT)
            )
        self.n_timesteps = n_timesteps
        self.time_mask = np.array(time_mask)
        if len(self.time_mask.shape) == 0:
            self.time_mask = self.time_mask[np.newaxis, np.newaxis]
            self.time_mask = np.tile(self.time_mask, [self.n_timesteps, self.proj_dim])
        elif len(self.time_mask.shape) == 1:
            if self.time_mask.size == self.n_timesteps:
                self.time_mask = self.time_mask[..., np.newaxis]
                self.time_mask = np.tile(self.time_mask, [1, self.proj_dim])
            elif self.time_mask.size == self.proj_dim:
                self.time_mask = self.time_mask[np.newaxis, ...]
                self.time_mask = np.tile(self.time_mask, [self.n_timesteps, 1])
            else:
                raise RuntimeError(
                    "Time mask shapes cannot be broadcasted to [n_timesteps x proj_dim]"
                )
        assert self.time_mask.shape[0] == n_timesteps
        assert self.time_mask.shape[1] == self.proj_dim

    def evaluate(self, dist, lagrange_var):
        self.mean, self.covar = uTransform(dist, self.transform_fcn)
        self.V_diag = tf.linalg.diag_part(self.covar)
        # mean: n_t x proj. dim
        # border: n_t x proj. dim
        # V_diag: n_t x proj. dim
        self.mass = self.std_normal.cdf(
            self.dir_vector * (self.mean - self.border) / tf.sqrt(self.V_diag)
        )
        # mass: n_t x proj. dim
        self.constraint = (self.mass - self.alpha) * self.time_mask

        return self.constraint * lagrange_var

    def evaluate_internal(self, dist):
        return tf.reduce_sum(self.evaluate(dist, self.lagrange_var))

    def update(self, x):
        self.lagrange_var = self.lagrange_var * tf.exp(
            self.lagrange_learning_rate * self.constraint
        )

    def info(self):
        print("Border1DConstraint constraint:\n")
        print("\tBorder :\t\t{}\n".format(self.border_np))
        print("\tDirection:\t\t{}\n".format(self.dir_vector_np))
        if hasattr(self, "lagrange_initial"):
            print("\tLagrange initial:\t\t{}\n".format(self.lagrange_initial))
        print("\tLagrange final:\t\t{}\n".format(self.lagrange_var.numpy))
        super(Border1DConstraint, self).info()
        print("\n")

    def plot(self, axs, tvec, color="k", linestyle="--"):
        if self.proj_dim == 1:
            axs = [axs]
        assert len(axs) == self.proj_dim
        for i in range(self.proj_dim):
            if self.time_mask.size == 1:
                axs[i].plot(
                    tvec, self.border_np[:, i], color=color, linestyle=linestyle
                )
            else:
                idx = self.time_mask[:, i] != 0
                axs[i].plot(
                    tvec[idx], self.border_np[idx, i], color=color, linestyle=linestyle
                )


class SmoothnessPenalty(Constraint):
    """Smoothness Penalty, can be used as an additional cost-function

    Parameters:
        dn_phi: n-th derivative to be considered
        n_outputs: number of output channels of the ProMP (typically number of dims)
        dt: system delta t
        priority: Can be used to prioritize the smoothness of specific dimensions
        scale: multiplier for the cost function (balances wrt. KL)
        floatT: tensorflow type used for variable initialisation
    """

    def __init__(
        self,
        dn_phi,
        n_outputs,
        dt,
        priority=[1.0],
        scale=1.0,
        floatT=tf.float64,
    ):
        with tf.variable_scope("SmoothnessPenalty"):
            super(SmoothnessPenalty, self).__init__(
                1.0,
                1.0,
                tf.identity,
                floatT=floatT,
            )
            self.lagrange_var = tf.constant(
                1.0,
                dtype=floatT,
            )  # Just a dummy, will not be used

        self.n_obs = n_outputs
        self.dt = dt
        self.scale = scale
        self.dn_phi = tf.constant(dn_phi, dtype=self.floatT)
        # Phi: n_basis x n_basis
        self.Phi = tf.transpose(dn_phi) @ dn_phi * self.dt
        self.priority = np.array(priority)
        if self.priority.size == 1:
            self.priority = self.priority * np.ones(self.n_obs)
        # Phi_stacked: (n_obs * n_basis) x (n_obs * n_basis)
        self.Phi_stacked = np.kron(np.diag(priority), self.Phi.numpy())
        self.Phi_stacked = tf.constant(self.Phi_stacked, dtype=self.floatT)

    def evaluate(self, dist):
        self.mean = dist.mean()
        self.covar = dist.covariance()
        # mean: n_t x proj. dim
        # border: n_t x proj. dim
        # V_diag: n_t x proj. dim
        self.E_smooth = tf.reduce_sum(
            tf.einsum("b...,bc,c...->...", self.mean, self.Phi_stacked, self.mean)
        ) + tf.linalg.trace(self.Phi_stacked @ self.covar)
        self.penalty = self.E_smooth * self.scale
        return self.penalty

    def update(self, x):
        pass

    def info(self):
        print("SmoothnessPenalty:\n")
        print("\tScale :\t\t{}\n".format(self.scale))
        print("\tPriority:\t\t{}\n".format(self.priority))
        print("\n")


class KLPenalty(Constraint):
    """KLPenalty adds another KL term and allows a transform fcn

    Parameters:
    prior: prior ProMP to compute KL against
    transform_fcn: (possibly nonlinear) transformation into penalty space
    scale: multiplier for the cost function (balances wrt. KL)
    use_uTransform: whether to use the unscented transformation with the given transform fcn.
    floatT: tensorflow type used for variable initialisation
    """

    def __init__(
        self,
        prior,
        transform_fcn=tf.identity,
        use_uTransform=False,
        scale=1.0,
        floatT=tf.float64,
    ):
        with tf.variable_scope("KLPenalty"):
            super(KLPenalty, self).__init__(
                1.0,
                1.0,
                tf.identity,
                floatT=floatT,
            )
            self.lagrange_var = tf.constant(
                1.0,
                dtype=floatT,
            )  # Just a dummy, will not be used
        self.scale = scale
        self.prior = prior
        self.transform_fcn = transform_fcn
        self.use_uTransform = use_uTransform

    def evaluate(self, dist):
        if self.use_uTransform:
            self.mean, self.covar = uTransform(dist, self.transform_fcn)
            kl_dist = tfp.distributions.MultivariateNormalFullCovariance(
                self.mean, self.covar
            )
        else:
            kl_dist = self.transform_fcn(dist)
        self.kl = tfp.distributions.kl_divergence(kl_dist, self.prior)
        self.penalty = self.kl * self.scale
        return self.penalty

    def update(self, x):
        pass

    def info(self):
        print("KLPenalty:\n")
        print("\tScale :\t\t{}\n".format(self.scale))
        print("\tPrior:\t\t{}\n".format(self.prior))
        print("\tTransform_fcn:\t\t{}\n".format(self.self.transform_fcn))
        print("\tuse_uTransform:\t\t{}\n".format(self.self.use_uTransform))
        print("\n")


class NonConvexConstraint(Constraint):
    """Non convex constraint, can be used to describe corner constraints

    Parameters:
        lagrange_learning_rate: learning rate for lagrange multiplier
        alpha: desired strictness of prob. constraint
        transform_fcn: nonlinear transformation into constraint space
        normal_vectors: normal vectors of the two walls
        intersection: corner point
        n_timesteps: number points in the timegrid
        time_mask: indicated at which timepoints the constraint is active (=1)
        floatT: tensorflow type used for variable initialisation
        lagrange_initial: initial value for lagrange multiplier
        vel_decay: unused?
    """

    def __init__(
        self,
        lagrange_learning_rate,
        alpha,
        transform_fcn,
        normal_vectors,
        intersection,
        n_timesteps,
        time_mask=1,
        floatT=tf.float64,
        lagrange_initial=1.0,
        vel_decay=0.1,
    ):
        self.lagrange_initial = lagrange_initial
        self.normal_vectors_np = np.array(normal_vectors)
        self.intersection_np = np.array(intersection)
        assert self.intersection_np.shape[-1] == self.normal_vectors_np.shape[-1]
        if (
            len(self.normal_vectors_np.shape) == 2
            and len(self.intersection_np.shape) == 1
        ):
            self.normal_vectors_np = self.normal_vectors_np[np.newaxis, ...]
            self.intersection_np = self.intersection_np[np.newaxis, :]
        elif (
            len(self.normal_vectors_np.shape) == 3
            and len(self.normal_vectors_np.shape) == 2
        ):
            # We have multiple constraints
            pass
        else:
            # Odd shapes
            raise RuntimeError(
                "Please check the shapes of the handed intersection and normal vector pairs"
            )
        with tf.variable_scope("NonConvexConstraint"):
            super(NonConvexConstraint, self).__init__(
                lagrange_learning_rate,
                alpha,
                transform_fcn,
                floatT=floatT,
            )
            self.normal_vectors = tf.constant(
                value=self.normal_vectors_np, dtype=floatT
            )  # n_con x 2 x cart. dim
            self.intersection = tf.constant(
                value=self.intersection_np, dtype=floatT
            )  # n_con x cart. dim
            self.lagrange_var = tf.squeeze(
                tf.Variable(
                    initial_value=lagrange_initial
                    * np.ones([n_timesteps, self.normal_vectors_np.shape[0]]),
                    dtype=floatT,
                )
            )  # n_ts x n_con or n_ts if n_con==1
            self.lagrange_var_vel = tf.squeeze(
                tf.zeros(
                    [n_timesteps, self.normal_vectors_np.shape[0]],
                    dtype=floatT,
                )
            )  # n_ts x n_con or n_ts if n_con==1
            self.std_normal = tfp.distributions.Normal(
                tf.constant(0.0, dtype=floatT), tf.constant(1.0, dtype=floatT)
            )
        self.n_timesteps = n_timesteps
        self.time_mask = np.array(time_mask)
        self.vel_decay = vel_decay
        if len(self.time_mask.shape) == 0:
            self.time_mask = self.time_mask[np.newaxis, np.newaxis]
            self.time_mask = np.tile(self.time_mask, [self.n_timesteps, 1])
        elif self.time_mask.size == self.n_timesteps:
            self.time_mask = self.time_mask[:, np.newaxis]
        else:
            assert self.time_mask.shape[0] == self.n_timesteps
            assert self.time_mask.shape[1] == self.normal_vectors_np.shape[0]

    def evaluate(self, dist, lagrange_var):
        self.mean, self.covar = uTransform(dist, self.transform_fcn)
        self.V_proj_diag = opt_einsum.contract(
            "cnd,...di,cni->...cn",
            self.normal_vectors,
            self.covar,
            self.normal_vectors,
            backend="tensorflow",
        )  # B x n_t x n_con x 2
        self.mass = self.std_normal.cdf(
            tf.einsum(
                "cnd,...cd->...cn",
                self.normal_vectors,
                self.mean[..., tf.newaxis, :] - self.intersection,
            )
            / tf.sqrt(self.V_proj_diag)
        )  # B x n_t x n_con x 2
        self.mass = tf.reduce_prod(self.mass, axis=-1)  # B x n_t x n_con
        self.constraint = tf.squeeze((self.mass - self.alpha) * self.time_mask)

        return self.constraint * lagrange_var

    def evaluate_internal(self, dist):
        return tf.reduce_sum(self.evaluate(dist, self.lagrange_var))

    def update(self, x):
        self.lagrange_var = self.lagrange_var * tf.exp(
            self.lagrange_learning_rate * self.constraint
        )

    def _rotate2dVector(self, vec, rotation):
        rotMat = np.array(
            [
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)],
            ]
        )
        return np.einsum("ij,...j->...i", rotMat, vec)

    def info(self):
        print("NonConvexConstraint:\n")
        print("\tNormal vectors:\t\t{}\n".format(self.normal_vectors_np))
        print("\tIntersection:\t\t{}\n".format(self.intersection))
        super(NonConvexConstraint, self).info()
        print("\n")

    def plot2D(self, ax):
        lvecs = self._rotate2dVector(self.normal_vectors_np, np.pi / 2)
        # n_con x 2 x n_cart
        for nvec, inter, lvec in zip(
            self.normal_vectors_np, self.intersection_np, lvecs
        ):
            for bidx in (np.array([True, False]), np.array([False, True])):
                if (
                    np.tensordot(
                        np.squeeze(lvec[bidx]), np.squeeze(nvec[~bidx]), axes=1
                    )
                    > 0
                ):
                    lv = lvec[bidx]
                else:
                    lv = -lvec[bidx]
                lv = np.squeeze(lv)
                nv = np.squeeze(nvec[bidx])
                linep1 = inter + 10 * lv
                linep2 = inter
                linepx = [linep1[0], linep2[0]]
                linepy = [linep1[1], linep2[1]]
                ax.plot(linepx, linepy, c="k")
                ax.arrow(
                    inter[0] + 1.5 * lv[0],
                    inter[1] + 1.5 * lv[1],
                    nv[0],
                    nv[1],
                    color="k",
                    width=0.02,
                )

    def plot3D(
        self, ax, n_mesh=100, color="r", ec="None", alpha=0.4, rstride=10, cstride=10
    ):
        raise NotImplementedError("plot3D for nonconvex missing")
        limits = np.array([[*ax.get_xlim()], [*ax.get_ylim()], [*ax.get_zlim()]])
        pos = np.linspace(limits[:, 0], limits[:, 1], num=n_mesh)
        for nvec, inter in zip(self.normal_vectors_np, self.intersection_np):
            cond = nvec != 0
            idx = cond.nonzero()[0][0]
            cond[...] = False
            cond[idx] = True
            mesh = np.zeros([3, n_mesh, n_mesh])
            a, b = np.meshgrid(pos[:, ~cond][:, 0], pos[:, ~cond][:, 1])
            mesh[~cond, ...] = np.stack((a, b))
            mesh[cond, ...] = (
                -1
                / nvec[cond]
                * np.einsum("n,n...->...", nvec[~cond], mesh[~cond, ...])
            )
            x = inter[0] + mesh[0, ...]
            y = inter[1] + mesh[1, ...]
            z = inter[2] + mesh[2, ...]
            ax.plot_surface(
                x,
                y,
                z,
                color=color,
                ec=ec,
                alpha=alpha,
                rstride=rstride,
                cstride=cstride,
            )
            ax.add_artist(
                Arrow3d(
                    [inter[0], nvec[0]],
                    [inter[1], nvec[1]],
                    [inter[2], nvec[2]],
                    arrowstyle="-|>",
                    color=color,
                    mutation_scale=10,
                    lw=1,
                )
            )
