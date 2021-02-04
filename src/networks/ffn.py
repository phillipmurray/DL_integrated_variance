from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, Conv1D
from tensorflow.keras.optimizers import Adam

from src.networks.ivmodel import IVModel

class FFNetwork(IVModel):
    def __init__(self, n_layers, h_dims=64, loss=None, batch_norm = False, **kwargs):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            if batch_norm:
                layers.append(BatchNormalization())
            layers.append(Dense(h_dims, activation='relu'))
        if batch_norm:
                layers.append(BatchNormalization())
        layers.append(Dense(h_dims, activation=None))
        if batch_norm:
                layers.append(BatchNormalization())
        layers.append(Dense(1, activation='softplus'))
        self.h_layers = layers
        self.loss = loss

        lr = kwargs.pop('lr', 0.001)
        self.optimizer = Adam(lr=lr)