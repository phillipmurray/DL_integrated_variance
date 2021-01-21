import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from scipy.special import factorial
from tqdm import tqdm_notebook
from src.utils import *
from tensorflow.keras.losses import KLDivergence

class JSDivergenceLoss:
    """Computes a Jensen-Shannon Divergence loss."""

    def __init__(self):
        pass

    def __call__(self,total_increments, integrated_var):
        #To DEBUG
        kl_div = KLDivergence()
        gen_sample = total_increments / tf.cast(tf.math.sqrt(integrated_var),tf.float32)
        gen_sample = tf.reshape(gen_sample, (gen_sample.shape[0], 1))
        z_sample = tf.random.normal(shape=gen_sample.get_shape(), dtype=tf.float32)
        cost = kl_div(gen_sample,z_sample) + kl_div(z_sample,gen_sample)
        cost /= 2
        return cost