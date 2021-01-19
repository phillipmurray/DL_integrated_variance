import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from scipy.special import factorial
from tqdm import tqdm_notebook
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
        gen_sample = tf.reshape(gen_sample, (gen_sample.shape[0], 1))
        cost = tf.reduce_mean(self.kernel(gen_sample, gen_sample))
        cost -= (self.length_scale/(1 + self.length_scale))**0.5 * tf.reduce_mean(2*tf.exp(-gen_sample**2/(2*(1 + self.length_scale))))
        #No need for last term since that is independent of the training data
        return cost
