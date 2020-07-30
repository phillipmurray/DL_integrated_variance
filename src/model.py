import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from scipy.special import factorial, factorial2

def gaussian_moment(p):
    if p % 2 == 0:
        return factorial2(p-1)
    else:
        return 0

class MomentLoss():
    def __init__(self, degree=4, p_norm=2, weights=None):
        self.degree = degree
        self.p_norm = p_norm
        self.weights = self._set_weights(weights)

    def _set_weights(self, weights):
        if weights is None:
            return 1.0
        elif weights == 'exponential':
            return np.geomspace(1.0, np.exp(-self.degree), num=self.degree)
        elif weights == 'factorial':
            return 1/factorial(np.linspace(0,self.degree, num=self.degree))

    def __call__(self, total_increments, integrated_var):
        gaussian_moments = K.constant([gaussian_moment(p) for p in range(1, self.degree+1)])
        z_sample = total_increments / K.sqrt(integrated_var)
        sample_moments = K.stack([K.mean(z_sample**p, axis=0) for p in range(1, self.degree+1)])
        moment_diffs = K.abs(sample_moments - gaussian_moments)
        return K.mean(self.weights * moment_diffs**self.p_norm)



class GaussianKernel():
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, x_1, x_2):
        return K.exp( -1/self.length_scale**2 * (x_1 - x_2)**2)




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

class MMDLoss():
    def __init__(self, kernel=gaussian_kernel_matrix, **kwargs):
        self.length_scale = kwargs.get('length_scale')
        self.kernel = self._set_kernel(kernel)

    def _set_kernel(self, kernel):
        if not isinstance(kernel, str):
            return kernel
        else:
            if kernel in ['rbf', 'gaussian']:
                if self.length_scale is not None:
                    return GaussianKernel(length_scale=self.length_scale)
                else:
                    return GaussianKernel()

    def __call__(self, total_increments, integrated_var):
        gen_sample = total_increments / K.sqrt(integrated_var)
        z_sample = tf.random.normal(shape=gen_sample.get_shape())
        cost = self.kernel(gen_sample, gen_sample)
        cost += self.kernel(z_sample, z_sample)
        cost -= 2*self.kernel(gen_sample, z_sample)
        #Ensure loss is non-negative
        cost = tf.where(cost > 0, cost, 0, name='value')
        return cost



class FFNetwork(Model):
    def __init__(self, n_layers, h_dims=64):
        layers = []
        for _ in range(n_layers):
            layers.append(Dense(h_dims, activation='relu'))
        layers.append(Dense(1, activation='softmax'))
        self.layers = layers
        self.model = Sequential(layers)

    def call(self, x):
        x = self.model(x)
        return x


