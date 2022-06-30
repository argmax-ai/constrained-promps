import os

import numpy as np
import scipy as sp
import tensorflow as tf


def tf_vectorize(A, name=None):
    # Vectorize Matrix A
    return tf.reshape(tf.transpose(A), [-1], name=name)


def sp_spd_inv(A):
    """Efficient inverse computation for symmetric positive definite matrices"""
    L_A = sp.linalg.cholesky(A, lower=True)
    rhs = sp.linalg.solve_triangular(L_A, np.eye(A.shape[0]), lower=True)
    return sp.linalg.solve_triangular(L_A.T, rhs)


def rotate2dVector(vec, rotation):
    """Rotate a given 2d-vector vec by rotation given in rad"""
    rotMat = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )
    return np.einsum("ij,...j->...i", rotMat, vec)


def safe_makedir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def tf_power_iteration(A, n_steps):
    # Tensorflow implementation of power iteration
    x_k = tf.ones(tf.shape(A)[:-1])

    for _ in range(n_steps):
        x_k1 = tf.einsum("...ij,...j->...i", A, x_k)
        x_k = x_k1 / tf.norm(x_k1, axis=-1)[..., tf.newaxis]

    return x_k
