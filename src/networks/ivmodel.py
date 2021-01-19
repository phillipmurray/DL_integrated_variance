import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from scipy.special import factorial
from tqdm import tqdm_notebook
from src.utils import *


class IVModel(Model):
    def __init__(self):
        super().__init__()

    def call(self, x):
        for layer in self.h_layers:
            x = layer(x)
        return x

    def _add_loss(self, loss):
        self.loss = loss

    def _set_lr(self, lr):
        if lr is not None:
            self.optimizer = Adam(lr=lr)

    def train_step(self, x_batch):
        total_increment = tf.squeeze(x_batch[:, -1, 0] - x_batch[:, 0, 0])
        with tf.GradientTape() as tape:
            int_var = self(x_batch)
            int_var = tf.squeeze(int_var)
            loss_value = self.loss(total_increment, int_var)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss_value

    def train(self, x_train, num_epochs, batch_size, lr=None, true_int_var=None, plotting=True):
        self._set_lr(lr)
        n_steps = x_train.shape[0] // batch_size
        x_train = create_train_dataset(x_train, batch_size)
        losses = []
        mses = []
        for epoch in range(num_epochs):
            with tqdm_notebook(total=n_steps, desc=f'Epoch {epoch+1} of {num_epochs}') as progress:
                for step, x_batch in enumerate(x_train):
                    progress.update()
                    loss_val = self.train_step(x_batch)
                    losses.append(loss_val.numpy())
            int_var = self(x_batch).numpy().squeeze()
            if true_int_var:
                mse_val = _mse_metric(true_int_var, int_var)
                mses.append(mse_val.numpy())
            if plotting:
                plot_hist(x_batch, int_var, true_int_var)

        self.history = {'loss': losses, 'mse': mses}
        return self.history

    def predict_iv(self, x):
        iv = self(x).numpy()
        return iv.squeeze()

    def predict_z(self, x):
        iv = self.predict_iv(x)
        z = (x[:,-1] - x[:,0])/np.sqrt(iv)
        return z