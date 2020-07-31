import numpy as np
import tensorflow as tf
from scipy.stats import norm
from scipy.special import factorial2
import matplotlib.pyplot as plt

def gaussian_moment(p):
    """Computes the moments of the standard normal distribution. Used
    when training via the method of moments.
    Args:
    p: an integer
    Returns:
    the pth moment of the standard Gaussian.
    """
    if p % 2 == 0:
        return factorial2(p-1)
    else:
        return 0

def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
    Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
    ValueError: if the inputs do no matched the specified dimensions.
    """
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, length_scale=1.0):
    """Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
    gaussians in the kernel.
    Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * length_scale)
    dist = compute_pairwise_distances(x, y)
    s = beta * tf.reshape(dist, (1, -1))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def plot_hist(x_batch, int_var, true_int_var=None, n_bins=20):

    total_increment = x_batch[:, -1] - x_batch[:, 0]
    gen_z = total_increment / np.sqrt(int_var)
    x = np.linspace(-3.0, 3.0, 100)
    if true_int_var is not None:
        plt.hist(true_int_var, bins=np.linspace(-3,3,n_bins), density=True, label='True')
    plt.hist(gen_z, bins=np.linspace(-3,3,n_bins), density=True, label='NN')
    plt.plot(x, norm.pdf(x, 0.0, 1.0))
    plt.legend()
    plt.show()