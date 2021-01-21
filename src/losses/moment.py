import numpy as np
import tensorflow as tf
from scipy.special import factorial
from src.utils import *


class MomentLoss:
    """Computes a loss using an Lp norm of the difference between between the moments
    of the training data and a standard Gaussian.
     """
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
        gaussian_moments = tf.constant([gaussian_moment(p) for p in range(1, self.degree+1)])
        z_sample = total_increments / tf.math.sqrt(integrated_var)
        sample_moments = tf.stack([tf.reduce_mean(z_sample**p, axis=0) for p in range(1, self.degree+1)])
        moment_diffs = tf.math.abs(sample_moments - gaussian_moments)
        return tf.reduce_mean(self.weights * moment_diffs**self.p_norm)