from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, MaxPooling1D, Dense
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm_notebook
import numpy as np
from src.networks.ivmodel import IVModel


class CNNetwork(IVModel):

    def __init__(self,n_timesteps, n_features, n_outputs,loss=None):
        super().__init__()
        self.timesteps = n_timesteps
        self.features = n_features
        self.outputs = n_outputs
        layers = []
        layers.append(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
        #layers.append(Conv1D(filters=64, kernel_size=3, activation='relu'))
        layers.append(Dropout(0.5))
        layers.append(MaxPooling1D(pool_size=2))
        layers.append(Flatten())
        layers.append(Dense(100, activation='relu'))
        layers.append(Dense(n_outputs, activation='softmax')) 
        self.h_layers = layers
        self.loss = loss
        self.optimizer = Adam(lr=0.001)
        self.compile(loss = loss)

