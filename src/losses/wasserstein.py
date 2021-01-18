import numpy as np


class WassersteinLoss:
    def __init__(self, mean = 0, std = 1):
        self.mean = mean
        self.std = std
        
    def __call__(self, model):

        gaussian_moments = tf.constant([gaussian_moment(p) for p in range(1, self.degree+1)])
        z_sample = total_increments / tf.math.sqrt(integrated_var)
        sample_moments = tf.stack([K.mean(z_sample**p, axis=0) for p in range(1, self.degree+1)])
        print(type(sample_moments))
        print(type(gaussian_moments))
        moment_diffs = tf.math.abs(sample_moments - gaussian_moments)
        return K.mean(self.weights * moment_diffs**self.p_norm)
