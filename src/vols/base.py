import numpy as np
import tensorflow as tf

class SemiMartingale:

    def __init__(self, **kwargs):
        """
            Generic class for a semimartingale process.

        """
        self.seed = kwargs.get('seed', None)

    
    def integrated_variance(self, return_mean=False):
        """
            Calculate the integrated variance from the paths

            Parameters:
            -----------
            return_mean: bool
                whether to calculate the mean integrated variance over
                all paths or not

            Returns:
            -----------
            numpy array of integrated variance for each path
        """
        int_var = np.sum(self.vol_paths**2, axis=1)/self.total_timesteps
        if return_mean:
            int_var = np.mean(int_var, axis=0)
        return int_var

    def get_increments(self):
        """
            Calculate the increments of the semimartingale process: X_t - X_{t-1}

            Returns:
            -----------
            numpy array of the increments
        """
        if self.paths is not None:
            return np.diff(self.paths, axis=-1)

    def reshape_for_rnn(self):
        """
            Reshape the paths into the correct shape for a tensorflow RNN

            Returns:
            -----------
            numpy array of shape (paths, timesteps, features)
        """
        if self.paths is not None:
            return np.reshape(self.paths, (self.n_paths, self.n_timesteps+1, 1))

    def get_total_increments(self):
        """
            Calculate the total increment of the path timeline: X_1 - X_0

            Returns:
            -----------
            numpy array of the total increment for each path
        """
        if self.paths is not None:
            return self.paths[:,-1] - self.paths[:,0]
    
    def get_log_returns(self):
        """
            Calculate log-returns. log(p_t) - log(p_t-1)

            Returns:
            -----------
            numpy array of the price log-returns for each path
        """
        return np.log(self.paths[:,1:]) - np.log(self.paths[:,:-1])

    def get_norm_increments(self):
        """
            Calculate the total increment normalised by the integrated variance:
            (X_1 - X_0) / sqrt(IV)

            Returns:
            -----------
            numpy array of normalised increments
        """
        if self.paths is not None:
            increments = self.get_total_increments()
            integrated_var = self.integrated_variance()
            return increments / np.sqrt(integrated_var)

    def _to_tensor(self, for_rnn=False):
        """
            Convert the numpy array paths to a tensorflow tensor

            Parameters:
            -----------
            for_rnn : bool
                Whether to reshape the paths for RNN

            Returns:
            -----------
            tensorflow tensor
        """
        if self.paths:
            if for_rnn:
                paths = self.reshape_for_rnn()
            else:
                paths = self.paths
            return tf.constant(paths)

    def add_noise(self, sd=0.5, seed=None):
        """
            Add Gaussian noise to the paths
            TODO: Change distribution of the noise - Gaussian is not a good
            model for market microstructure noise

            Parameters:
            -----------
            sd : float
                Standard deviation of the noise
            seed : int
                random seed to use

            Returns:
            -----------
            numpy array of the paths with added noise
        """
        self.seed = seed
        np.random.seed(seed)
        if self.paths is not None:
            noise = np.random.normal(scale=sd, size=self.paths.shape)
        noisy_paths = self.paths + noise
        return noisy_paths
