import numpy as np
import tensorflow as tf


class Wasserstein2Loss:
    """2-Wasserstein distance  between 2 Normal Distributions
    Remark: 2-Wasserstein distance is an upper bound of 
    1-Wasserstein distance.
    """
    def __init__(self, mean = 0, variance = 1):
        self.mean = mean
        self.variance = variance
        
    def __call__(self, total_increments, integrated_var):
        gen_sample = total_increments / tf.cast(tf.math.sqrt(integrated_var),tf.float32)
        gen_sample = tf.reshape(gen_sample, (gen_sample.shape[0], 1))
        sample_mean = tf.reduce_mean(gen_sample)
        sample_variance = tf.math.reduce_std(gen_sample)**2
        cost = tf.math.abs(sample_mean -self.mean)
        cost += self.variance + sample_variance 
        cost -= 2*tf.math.sqrt(sample_variance*self.variance)
        print(cost)
        return cost