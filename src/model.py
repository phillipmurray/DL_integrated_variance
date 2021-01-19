import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from scipy.special import factorial
from tqdm import tqdm_notebook
from src.utils import *



class WassersteinLoss:

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, total_increments, integrated_var):
        gen_sample = total_increments / tf.math.sqrt(integrated_var)
        gen_sample = tf.reshape(gen_sample, (gen_sample.shape[0], 1))
        #TODO: finish
        pass








