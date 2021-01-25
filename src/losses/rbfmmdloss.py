import tensorflow as tf
from src.utils import *


class RBFMMDLoss:
    """Computes a MMD loss using the Gaussian Radial Basis Kernel
     using the analytic form.
     """
    def __init__(self, kernel=gaussian_kernel_matrix, length_scale=1.0):
        self.length_scale = length_scale
        self.kernel = GaussianKernel(length_scale=self.length_scale)

    def __call__(self, total_increments, integrated_var):
        gen_sample = total_increments / tf.math.sqrt(integrated_var)
        n = gen_sample.shape[0]
        gen_sample = tf.reshape(gen_sample, (n, 1))
        cost = (self.length_scale/(2 + self.length_scale))**0.5
        cost += 1/n/(n-1)*tf.reduce_sum(self.kernel(gen_sample, gen_sample))
        cost -= 1/(n-1)
        cost -= 2*(self.length_scale/(1 + self.length_scale))**0.5 * tf.reduce_mean(tf.exp(-gen_sample**2/(2*(1 + self.length_scale))))
        return cost
