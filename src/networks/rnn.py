import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from scipy.special import factorial
from tqdm import tqdm_notebook
from src.utils import *
from src.networks.ivmodel import IVModel


class RNNNetwork(IVModel):
    def __init__(self, n_layers, h_dims=64, recurrent_type='rnn', loss=None, **kwargs):
        super().__init__()
        layers = []
        h_layer = SimpleRNN if recurrent_type == 'rnn' else LSTM
        for _ in range(n_layers):
            layers.append(h_layer(h_dims, return_sequences=True))
        layers.append(h_layer(h_dims, return_sequences=False))
        for _ in range(n_layers):
            layers.append(Dense(h_dims, activation='relu'))
        layers.append(Dense(1, activation='softplus'))
        self.h_layers = layers
        self.loss = loss

        lr = kwargs.pop('lr', 0.001)
        self.optimizer = Adam(lr=lr)