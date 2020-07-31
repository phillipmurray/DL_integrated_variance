import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scipy.special import factorial
from tqdm import tqdm_notebook
from src.utils import *

K.set_floatx('float64')


class MomentLoss():
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
        gaussian_moments = K.constant([gaussian_moment(p) for p in range(1, self.degree+1)])
        z_sample = total_increments / K.sqrt(integrated_var)
        sample_moments = K.stack([K.mean(z_sample**p, axis=0) for p in range(1, self.degree+1)])
        moment_diffs = K.abs(sample_moments - gaussian_moments)
        return K.mean(self.weights * moment_diffs**self.p_norm)

class MMDLoss():
    def __init__(self, kernel=gaussian_kernel_matrix, **kwargs):
        self.length_scale = kwargs.get('length_scale')
        self.kernel = self._set_kernel(kernel)

    def _set_kernel(self, kernel):
        if not isinstance(kernel, str):
            return kernel
        else:
            pass

    def __call__(self, total_increments, integrated_var):
        gen_sample = total_increments / K.sqrt(integrated_var)
        gen_sample = K.reshape(gen_sample, (gen_sample.shape[0], 1))
        z_sample = tf.random.normal(shape=gen_sample.get_shape(), dtype='float64')
        cost = tf.reduce_mean(self.kernel(gen_sample, gen_sample))
        cost += tf.reduce_mean(self.kernel(z_sample, z_sample))
        cost -= tf.reduce_mean(2*self.kernel(gen_sample, z_sample))
        #Ensure loss is non-negative
        cost = tf.where(cost > 0, cost, 0, name='value')
        return cost



@tf.function
def create_train_dataset(x_train, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset


def _mse_metric(var_true, var_pred):
    return tf.reduce_mean((var_true - var_pred)**2)


class FFNetwork(Model):
    def __init__(self, n_layers, h_dims=64, loss=None):
        super(FFNetwork, self).__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(Dense(h_dims, activation='relu'))
        layers.append(Dense(1, activation='softplus'))
        self.h_layers = layers
        self.loss = loss
        self.optimizer = Adam(lr=0.001)

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
        total_increment = x_batch[:, -1] - x_batch[:, 0]
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

