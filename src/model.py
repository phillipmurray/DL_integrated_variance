import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from scipy.special import factorial
from tqdm import tqdm_notebook
from src.utils import *


class MomentLoss:
    """Computes a loss using an Lp norm of the difference between between the moments
    of the training data and a standard Gaussian.
     """
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
        gaussian_moments = tf.constant([gaussian_moment(p) for p in range(1, self.degree+1)])
        z_sample = total_increments / tf.math.sqrt(integrated_var)
        sample_moments = tf.stack([tf.reduce_mean(z_sample**p, axis=0) for p in range(1, self.degree+1)])
        moment_diffs = tf.math.abs(sample_moments - gaussian_moments)
        return tf.reduce_mean(self.weights * moment_diffs**self.p_norm)

class MMDLoss:
    """Computes a MMD loss using a kernel function (the Gaussian Radial Basis Kernel
    by default) using samples from the training data and from a standard Gaussian.
     """
    def __init__(self, kernel=GaussianKernel, **kwargs):
        self.length_scale = kwargs.get('length_scale')
        self._set_kernel(kernel)

    def _set_kernel(self, kernel):
        if not isinstance(kernel, str):
            self.kernel = kernel(length_scale=self.length_scale)
        else:
            pass

    def __call__(self, total_increments, integrated_var):
        gen_sample = total_increments / tf.math.sqrt(integrated_var)
        gen_sample = tf.reshape(gen_sample, (gen_sample.shape[0], 1))
        z_sample = tf.random.normal(shape=gen_sample.get_shape(), dtype=tf.float32)
        cost = tf.reduce_mean(self.kernel(gen_sample, gen_sample))
        cost += tf.reduce_mean(self.kernel(z_sample, z_sample))
        cost -= tf.reduce_mean(2*self.kernel(gen_sample, z_sample))
        #Ensure loss is non-negative
        cost = tf.where(cost > 0, cost, 0, name='value')
        return cost

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


class WassersteinLoss:

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, total_increments, integrated_var):
        gen_sample = total_increments / tf.math.sqrt(integrated_var)
        gen_sample = tf.reshape(gen_sample, (gen_sample.shape[0], 1))
        #TODO: finish
        pass



@tf.function
def create_train_dataset(x_train, batch_size):
    """Creates a batched dataset from a given numpy array
    """
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset

@tf.function
def _mse_metric(var_true, var_pred):
    """Calculates the MSE between the predicted integrated variance and
    true integrated variance"""
    return tf.reduce_mean((var_true - var_pred)**2)





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


class FFNetwork(IVModel):
    def __init__(self, n_layers, h_dims=64, loss=None, **kwargs):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(Dense(h_dims, activation='relu'))
        layers.append(Dense(h_dims, activation=None))
        layers.append(Dense(1, activation='softplus'))
        self.h_layers = layers
        self.loss = loss

        lr = kwargs.pop('lr', 0.001)
        self.optimizer = Adam(lr=lr)




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