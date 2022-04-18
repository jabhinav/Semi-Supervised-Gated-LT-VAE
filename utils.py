import tensorflow as tf
import numpy as np
from tensorflow_probability.python.distributions import Laplace, Normal
from tensorflow_probability.python.distributions.kullback_leibler import kl_divergence


def sample_gumbel_np(size, eps=1e-10):
    """ Sample from Gumbel(0, 1)"""
    u = np.random.random(size)
    g = -np.log(-np.log(u + eps) + eps)
    return g


def softmax_np(X, temperature=1.0, axis=1):
    """
    Compute the softmax of each element along an axis of X.
    """
    X = X / float(temperature)
    return np.exp(X) / np.sum(np.exp(X), axis=axis, keepdims=True)


def gumbel_softmax_np(logits, temperature=1, is_prob=False, ):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    if is_prob:
        logits = np.log(logits)
    y = logits + sample_gumbel_np(np.shape(logits))
    return softmax_np(y, temperature=temperature)


def sample_normal_np(mu, std, latent_dim):
    epsilon = np.random.normal(loc=0.0, scale=1.0, size=std.shape)
    z = mu + np.multiply(std, epsilon)
    return z


def multi_sample_normal_np(mu, std, latent_dim, k=100):
    samples = []
    for _ in range(k):
        z = sample_normal_np(mu, std, latent_dim)
        samples.append(z)
    return samples


def multi_sample_normal_tf(locs, scales, k):
    samples = []
    for _ in tf.range(k):
        epsilon = tf.random.normal(tf.shape(scales), mean=0.0, stddev=1.0, )
        z = locs + tf.math.multiply(scales, epsilon)
        samples.append(z)
    return samples


def kl_divergence_unit_gaussian(mu, log_sigma_sq, mean_batch=True, name='kl_divergence_unit_gaussian'):
    # KL divergence between a multivariate Gaussian distribution with diagonal covariance and an
    # isotropic Gaussian distribution
    latent_loss = -0.5 * tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq), axis=1)
    if mean_batch:
        latent_loss = tf.reduce_mean(latent_loss, axis=0)
    return latent_loss


def kl_divergence_gaussian(mu1, log_sigma_sq1, mu2, log_sigma_sq2, mean_batch=True, name='kl_divergence_gaussian'):
    # Ref: https://github.com/Era-Dorta/tf_mvg/blob/master/mvg_distributions/kl_divergence.py
    # Ref: https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    # KL divergence between two multivariate Gaussian distributions with diagonal covariance
    # All inputs must be matrices of [batch size, number of features]
    k = tf.cast(tf.shape(mu1), mu1.dtype)[1]  # Number of features
    kl_div = 0.5 * (
            # log(|sigma2|/|sigma1|) = log(prod_of_diag(sigma2)) - log(prod_of_diag(sigma1))
            #                        = sum_of_log_of_diag(sigma2) - sum_of_log_of_diag(sigma1)
            tf.reduce_sum(log_sigma_sq2, axis=1) - tf.reduce_sum(log_sigma_sq1, axis=1) - k +  # -k
            # trace(inv(sigma2) sigma1) <- div. the entries of sigma1 with that of sigma2 and then sum them to get trace
            tf.reduce_sum(tf.exp(log_sigma_sq1 - log_sigma_sq2), axis=1) +
            # (mu1 - mu2)^T inv(sigma2) (mu1 - mu2) <- divide the entries of (mu1 - mu2)^2 with entries of sigma2 (diag)
            tf.reduce_sum(tf.math.multiply((mu1 - mu2) ** 2, tf.exp(-log_sigma_sq2)), axis=1)
            #  equivalent exp. of tf.einsum('bi,bi->b', (mu1 - mu2) ** 2, tf.exp(-log_sigma_sq2)),
            # 'bi,bi->b' says for each element b in the batch, take the dot-prod i.e. i,i->
            # (However this is giving unicode error, therefore using equivalent exp. above)
    )
    if mean_batch:
        return tf.reduce_mean(kl_div, axis=0)
    else:
        return kl_div


def get_transn_loss(un_prev_encode_y, un_curr_encode_y):
    """
    Returns: 1 - c_{t-1} * c_t / ( ||c_{t-1}|| * ||c_t|| ) instead of 1 - c_{t-1} * c_t / max( ||c_{t-1}||, ||c_t|| )

    """

    y_penalty = tf.math.l2_normalize(un_curr_encode_y, dim=1) * tf.math.l2_normalize(un_prev_encode_y, dim=1)
    y_penalty = tf.reduce_sum(y_penalty, axis=1)
    transn_loss = 1. - y_penalty
    transn_loss = tf.reduce_mean(transn_loss)
    return transn_loss


def img_log_likelihood(recon, xs):
    return tf.reduce_sum(Laplace(recon, tf.ones_like(recon)).log_prob(xs), axis=[1,2,3])


def get_gaussian_kl_div(locs_q, scale_q, locs_p=None, scale_p=None):
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = tf.zeros_like(locs_q)
    if scale_p is None:
        scale_p = tf.ones_like(scale_q)

    dist_q = Normal(locs_q, scale_q)
    dist_p = Normal(locs_p, scale_p)
    return tf.reduce_sum(kl_divergence(dist_q, dist_p), axis=-1)