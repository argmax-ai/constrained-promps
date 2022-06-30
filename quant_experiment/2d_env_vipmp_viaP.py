# 2-D environment with randomly placed temporally unbound waypoints

import copy
import datetime
import os
import pickle
import time
from collections.abc import Container

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from box import Box
from trajectory_env import TrajectoryEnv

from opt_pmp_utils.constraints import OneTimeWaypoint, SmoothnessPenalty
from opt_pmp_utils.cpmp import CProMP
from opt_pmp_utils.utils import safe_makedir


def generate_vipmp_callback(model, env, path):
    safe_makedir(path)

    def callback(opt_res):
        cwd = os.getcwd()
        os.chdir(path)
        fig, axs = env.compare_show(10, model, model.marginals)
        axs[3].set_title("Adapted (CPMP)")
        axs[3].text(
            -3.0, 3.5, f"iter={model.lam_iters}, viol.={model.violation_hist[-1]}"
        )
        fig.savefig(f"vipmp_iter_{model.lam_iters}.png")
        plt.close(fig)
        with h5py.File(f"vipmp_iter_{model.lam_iters}.hdf5", mode="w") as f:
            f.create_dataset("opt_res.position", data=opt_res.position)
        os.chdir(cwd)

    return callback


def main():
    # tf.random.set_seed(777)
    # np.random.seed(777)
    n_experiments = 3  # Number of experiments to run
    nt = 40  # Number of points in the timegrid
    nbasis_pmp = 20  # Number of basis functions for the ProMP
    n_dim = 2  # only 2 is supported
    n_paths_violations = int(
        1e5
    )  # Number of paths sampled to test for violations of the constraints
    n_t_via_points = [
        1,
        2,
        3,
    ]  # Number of sampled temporally unbound via-points, can be list of ints or int
    n_via_points = 2  # Number of via-points for the initial ProMP (only 2 is supported)
    via_point_var = (
        1e-3  # Variance for conditioning the original ProMP on the via-points
    )
    pmp_prior_var = 0.8  # Prior ProMP variance
    tVia_margin_min = 0.05  # Minimum radius of the via-points
    tVia_margin_max = 0.2  # Maximum radius of the via-points
    smooth_scale = 0.0  # Scale of the smoothness penalty (0 = off)
    std_distance = 1.1  # unused (?)
    window_margin = 0  # Window margin of the temporally unbound via-point
    t_via_y_min = -0.7  # Minimum y-coordinate of the sampled via-points
    t_via_y_max = 0.7  # Minimum y-coordinate of the sampled via-points
    # VIPMP Parameter
    n_iter_cpmp = 10  # Maximum iterations of the CPMP algorithm
    n_sub_iter_cpmp = 500  # Maximum iterations of the sub L-BFGS algorithm
    callback_freq_vipmp = 5  # Frequency of saving an intermediate checkpoint
    lagrange_learning_rate = 1.0  # Learning rate of the lagrange multipliers
    lagrange_initial = 20.0  # Initial value of the lagrange multipliers
    alpha = 1e-3  # Strictness of the constraints
    ##################
    experiment_folder = f"output/experiments_viaP/"

    for i in range(len(n_t_via_points) if isinstance(n_t_via_points, Container) else 1):
        for ii in range(n_experiments):
            exp_name = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
            env = TrajectoryEnv(
                n_dim,
                nt,
                nbasis_pmp,
                pmp_prior_var=pmp_prior_var,
                n_obstacles=0,
                n_t_via_points=n_t_via_points[i]
                if isinstance(n_t_via_points, Container)
                else n_t_via_points,
                n_via_points=n_via_points,
                tVia_margin_min=tVia_margin_min,
                tVia_margin_max=tVia_margin_max,
                via_point_var=via_point_var,
                std_distance=std_distance,
                t_via_y_min=t_via_y_min,
                t_via_y_max=t_via_y_max,
            )

            # CPMP CODE
            with tf.device("CPU"):
                model = CProMP(n_dim, basis_fun=env.bfun, const=[])
                # Constraints
                for t_via in env.t_via_points:
                    model.add_constraint(
                        OneTimeWaypoint(
                            lagrange_learning_rate,
                            alpha,
                            tf.identity,
                            t_via["point"],
                            t_via["margin"],
                            nt,
                            lagrange_initial=lagrange_initial,
                            window_margin=1,
                        )
                    )
                model.add_penalty(
                    SmoothnessPenalty(
                        model.basis_fun.ddX,
                        n_dim,
                        model.basis_fun.dt[0],
                        priority=np.ones(n_dim),
                        scale=smooth_scale,
                    )
                )
                cb = generate_vipmp_callback(
                    model, env, experiment_folder + exp_name + "/"
                )
                model.set_prior(env.pmp.m0, env.pmp.V0, env.pmp.Q0)
                x0 = tf.constant(model.get_initial(True))
                try:
                    start = time.time()
                    opt_res = model.fit(
                        x0,
                        max_iters=int(n_iter_cpmp),
                        sub_iters=int(n_sub_iter_cpmp),
                        callback=cb,
                        callback_freq=callback_freq_vipmp,
                        n_paths_violations=n_paths_violations,
                        f_improved_tolerance=1e-2,
                    )
                    runtime = time.time() - start
                except Exception as e:
                    print(e)
                    continue
            print("Converged: {}".format(opt_res.converged))
            print("iterations: {}".format(model.opt_iters))
            print("lagrange_updates: {}".format(model.lam_iters))
            print("evaluations: {}".format(model.opt_evals))
            x_opt = opt_res.position
            M, log_diag_L_V, off_diag_L_V = model._unroll_var()(x_opt)
            model.graph(M, log_diag_L_V, off_diag_L_V)

            paths = model.sample(n_paths_violations)
            violations = tf.zeros([n_paths_violations], dtype=tf.bool)
            for const in model.const:
                violations = tf.math.logical_or(violations, const.get_violations(paths))
            perc_violations = np.sum(violations) / n_paths_violations
            kl_2_prior = tfp.distributions.kl_divergence(model.w_dist, env.w_dist)

            fig, axs = env.compare_show(10, model, model.marginals)
            axs[3].set_title("Adapted (CPMP)")
            axs[3].text(
                -3.0, 3.5, f"iter={model.lam_iters}, viol.={model.violation_hist[-1]}"
            )
            axs[3].text(-3.0, 3.0, f"kl={kl_2_prior}")
            fig.savefig(experiment_folder + exp_name + "/" + f"vipmp_final.png")
            plt.close(fig)

            save = Box(
                nt=nt,
                nbasis_pmp=nbasis_pmp,
                n_dim=n_dim,
                n_paths_violations=n_paths_violations,
                env=env,
                loss_hist=model.loss_hist,
                violation_hist=model.violation_hist,
                kl_hist=model.kl_hist,
                lam_hist=model.lam_hist,
                const_hist=model.const_hist,
                penalty_hist=model.penalty_hist,
                opt_res=opt_res,
                constraints=model.const,
                penalties=model.penalty,
                lam_iters=model.lam_iters,
                opt_iters=model.opt_iters,
                runtime=runtime,
                violations=perc_violations,
                kl_2_prior=kl_2_prior,
                n_iter_cpmp=n_iter_cpmp,
                n_sub_iter_cpmp=n_sub_iter_cpmp,
                callback_freq_vipmp=callback_freq_vipmp,
                lagrange_learning_rate=lagrange_learning_rate,
                alpha=alpha,
            )

            with open(
                experiment_folder + exp_name + "/vipmp_final_save", mode="wb"
            ) as f:
                pickle.dump(save, f)
            del save
            # CPMP CODE


if __name__ == "__main__":
    main()
