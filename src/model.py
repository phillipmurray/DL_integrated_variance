import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scipy.special import factorial, factorial2

tf.keras.backend.set_floatx('float64')


def gaussian_moment(p):
    if p % 2 == 0:
        return factorial2(p-1)
    else:
        return 0

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



class GaussianKernel():
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, x_1, x_2):
        return K.exp( -1/self.length_scale**2 * (x_1 - x_2)**2)




def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
    Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
    ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, length_scale=1.0):
    """Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
    gaussians in the kernel.
    Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * length_scale)
    dist = compute_pairwise_distances(x, y)
    s = beta * tf.reshape(dist, (1, -1))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

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


class FFNetwork(Model):
    def __init__(self, n_layers, h_dims=64, loss=None):
        super(FFNetwork, self).__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(Dense(h_dims, activation='relu'))
        layers.append(Dense(1, activation='softplus'))
        self.h_layers = layers
        self.loss = loss

    def call(self, x):
        for layer in self.h_layers:
            x = layer(x)
        return x

    def _add_loss(self, loss):
        self.loss = loss

    def train(self, x_train, num_epochs, batch_size, lr):
        optimizer = Adam(lr=lr)
        x_train = create_train_dataset(x_train, batch_size)

        @tf.function
        def train_step(x_batch):
            total_increment = x_batch[:,-1] - x_batch[:,0]
            with tf.GradientTape() as tape:
                int_var = self.call(x_batch)
                int_var = K.reshape(int_var, x_batch.shape[0     ])
                loss_value = self.loss(total_increment, int_var)
            grads = tape.gradient(loss_value, self.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return loss_value

        for epoch in range(num_epochs):
            for step, x_batch in enumerate(x_train):
                train_step(x_batch)


