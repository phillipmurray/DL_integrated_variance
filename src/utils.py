import numpy as np
import tensorflow as tf
from scipy.stats import norm
from scipy.special import factorial2
import matplotlib.pyplot as plt

def gaussian_moment(p):
    """
        Computes the moments of the standard normal distribution. Used
        when training via the method of moments.

        Parameters:
        -----------
        p: int

        Returns:
        -----------
        the pth moment of the standard Gaussian.
    """
    if p % 2 == 0:
        return factorial2(p-1)
    else:
        return 0

def compute_pairwise_distances(x, y):
    """
        Computes the squared pairwise Euclidean distances between x and y.

        Parameters:
        -----------
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]

        Returns:
        -----------
        a distance matrix of dimensions [num_x_samples, num_y_samples].

        Raises:
        -----------
        ValueError: if the inputs do no matched the specified dimensions.
    """
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, length_scale=1.0):
    """
        Computes a Guassian Radial Basis Kernel between the samples of x and y.
        We create a sum of multiple gaussian kernels each having a width sigma_i.

        Parameters:
        -----------
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.

        Returns:
        -----------
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * length_scale)
    dist = compute_pairwise_distances(x, y)
    s = beta * tf.reshape(dist, (1, -1))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

class GaussianKernel:
    def __init__(self, length_scale):
        self.length_scale = length_scale
    def __call__(self, x, y):
        return gaussian_kernel_matrix(x, y, self.length_scale)



def plot_hist(x_batch, int_var, true_int_var=None, n_bins=20):

    total_increment = x_batch[:, -1] - x_batch[:, 0]
    gen_z = total_increment / np.sqrt(int_var)
    x = np.linspace(-3.0, 3.0, 100)
    if true_int_var is not None:
        plt.hist(true_int_var, bins=np.linspace(-3,3,n_bins), density=True, label='True')
    plt.hist(gen_z, bins=np.linspace(-3,3,n_bins), density=True, label='NN')
    plt.plot(x, norm.pdf(x, 0.0, 1.0))
    plt.legend()
    plt.show()

def plot_loss(losses):
    plt.plot(losses,'r--')
    plt.grid()
    plt.title("Loss During Training")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.show()
    
def set_plot_params():
    """Useful function to set parameters for inline matplotlib plots
    in Jupyter notebooks.
    """
    plt.rcParams.update({
        'mathtext.fontset': 'cm',
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.,
        'axes.labelpad': 2.,
        'figure.dpi': 100,
        'figure.figsize': (6, 6),
        'legend.frameon': False,
        'animation.html': 'html5'
    })




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


