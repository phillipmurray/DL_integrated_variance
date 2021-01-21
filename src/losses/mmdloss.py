import tensorflow as tf

from src.utils import *


class MMDLoss:
    """Computes a MMD loss using a kernel function (the Gaussian Radial Basis Kernel
    by default) using samples from the training data and from a standard Gaussian.
     """
    def __init__(self, kernel=GaussianKernel, **kwargs):
        self.length_scale = kwargs.get('length_scale')
        self._set_kernel(kernel)

    def _set_kernel(self, kernel):
        if not isinstance(kernel, str):
            self.kernel = kernel(length_scale=self.length_scale)
        else:
            pass

    def __call__(self, total_increments, integrated_var):
        gen_sample = total_increments / tf.math.sqrt(integrated_var)
        gen_sample = tf.reshape(gen_sample, (gen_sample.shape[0], 1))
        z_sample = tf.random.normal(shape=gen_sample.get_shape(), dtype=tf.float32)
        cost = tf.reduce_mean(self.kernel(gen_sample, gen_sample))
        cost += tf.reduce_mean(self.kernel(z_sample, z_sample))
        cost -= tf.reduce_mean(2*self.kernel(gen_sample, z_sample))
        #Ensure loss is non-negative
        cost = tf.where(cost > 0, cost, 0, name='value')
        return cost