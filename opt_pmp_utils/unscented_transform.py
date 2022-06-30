import tensorflow as tf


def _uTransform(
    normalDist, transform, tf_sqrt=tf.linalg.sqrtm, alpha=1e-1, beta=2.0, kappa=-1
):
    Ndim = tf.squeeze(normalDist.event_shape_tensor())  # Event shape
    NdimF = tf.cast(Ndim, normalDist.dtype)
    lam = (alpha**2 + kappa) * NdimF  # some scaling
    # Compute weighting vectors for computing mean and covariance from samples
    # See
    weigthsMean0 = tf.ones([1], dtype=normalDist.dtype) * lam / (NdimF + lam)
    weigthsMean1 = tf.ones([2 * Ndim], dtype=normalDist.dtype) / 2 / (NdimF + lam)
    weigthsMean = tf.concat((weigthsMean0, weigthsMean1), axis=0)
    weightsCovariance0 = tf.ones([1], dtype=normalDist.dtype) * lam / (NdimF + lam) + (
        1 - alpha**2 + beta
    )
    weightsCovariance1 = tf.ones(2 * Ndim, dtype=normalDist.dtype) / 2 / (NdimF + lam)
    weightsCovariance = tf.concat((weightsCovariance0, weightsCovariance1), axis=0)

    # Distance from mean based on square root of the covariance
    # sigM: n x n x B
    # Is already transposed so we have col vectors in first dimension
    n_batch_dim = len(normalDist.batch_shape)
    sigM = tf.transpose(
        tf_sqrt((NdimF + lam) * normalDist.covariance()),
        perm=[n_batch_dim + 1, *range(n_batch_dim), n_batch_dim],
    )

    # All samples are either mean or mean +/- sigM -> repeat mean
    tile_arg = tf.concat(
        [
            tf.ones_like(normalDist.event_shape_tensor()) * (2 * Ndim + 1),
            tf.ones_like(normalDist.batch_shape_tensor()),
            tf.ones_like(normalDist.event_shape_tensor()),
        ],
        axis=0,
    )
    # Mean: (2n+1) x B x n
    Mean = tf.tile(normalDist.mean()[tf.newaxis, ...], tile_arg)
    # samplePoints: (2n+1) x B x n
    samplePoints = Mean + tf.concat(
        (tf.zeros_like(normalDist.mean())[tf.newaxis, ...], sigM, -sigM), 0
    )
    transformedPoints = transform(samplePoints)  # transformedPoints: (2n+1) x B x o
    # Batch matmul [transformedMean: B x o]
    transformedMean = tf.einsum("m,m...->...", weigthsMean, transformedPoints)
    meanDiff = transformedPoints - transformedMean  # meanDiff: (2n+1) x B x o
    # Weighted outer product over sample points
    Vy = tf.einsum(
        "m...o,mn,n...l->...ol",
        meanDiff,
        tf.linalg.tensor_diag(weightsCovariance),
        meanDiff,
    )  # Vy: B x o x o
    return transformedMean, Vy, transformedPoints


def _uTransform_mV(
    normalDist, transform, tf_sqrt=tf.linalg.sqrtm, alpha=1e-1, beta=2.0, kappa=-1
):
    m, V, _ = _uTransform(
        normalDist,
        transform,
        tf_sqrt=tf_sqrt,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
    return m, V


def uTransform(normalDist, transform, alpha=1e-1, beta=2.0, kappa=-1):
    return _uTransform_mV(
        normalDist,
        transform,
        tf_sqrt=tf.linalg.sqrtm,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )


def uTransform_cholesky(normalDist, transform, alpha=1e-1, beta=2.0, kappa=-1):
    return _uTransform_mV(
        normalDist,
        transform,
        tf_sqrt=tf.linalg.cholesky,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )


def uTransform_mViP(normalDist, transform, alpha=1e-1, beta=2.0, kappa=-1):
    return _uTransform(
        normalDist,
        transform,
        tf_sqrt=tf.linalg.sqrtm,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
