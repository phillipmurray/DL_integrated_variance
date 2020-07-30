import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
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



class MMDLoss():
    def __init__(self, kernel=None, **kwargs):
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
        z_sample = tf.random.normal(shape=gen_sample.shape)
        return self.kernel()