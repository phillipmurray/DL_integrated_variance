import numpy as np
import tensorflow as tf
from src.vols.expOU import ExpOU

class SemiMartingale:

    def __init__(self, X_0, vol, **kwargs):
        """
            Generic class for a semimartingale process.

            Parameters:

            X_0 : float
                the initial value of the semimartingale process
            vol : str / np.array
                the stochastic volatility process driving the semimartingale.
                Can either be a string or separately generated volatility process.
        """
        self.X_0 = X_0
        self.vol = vol
        self.seed = None
        self.n_paths = None
        self.total_timesteps = None
        self.n_timesteps = None
        self.paths = None

    def _generate_vol(self, vol, n_paths, total_timesteps, n_timesteps=None, seed=None, **kwargs):
        """
            Generate the volatility process over the number of timesteps

            Parameters:
            -----------
            vol : float / str / np.array
                the vol process to generate
            n_paths : int
                the number of paths to generate
            total_timesteps : int
                the total number of timesteps that the volatility process is generated over for
                a single day - the frequency of observations
            n_timesteps : int
                the number of timesteps to generate at this frequency
            seed : int
                random seed to be used

            Returns:
            -----------
            numpy array of volatility
        """
        self.seed = seed
        np.random.seed(seed)
        if n_timesteps is None:
            n_timesteps = total_timesteps
        if isinstance(vol, float):
            return vol * np.ones((n_paths, n_timesteps))
        elif isinstance(vol, np.ndarray):
            if vol.shape != (n_paths, n_timesteps):
                raise ValueError("Vol must have same dimensions as path")
            else:
                return vol
        elif isinstance(vol, str):
            if vol == 'expou':
                theta, beta = kwargs.get('theta', 1.0), kwargs.get('beta', 0.2)
                return ExpOU(theta, beta).generate(n_paths, total_timesteps, n_timesteps, seed)


    def generate(self, n_paths, total_timesteps, n_timesteps=None, seed=None, **kwargs):
        """
            Generate the semimartingale process

            Parameters:
            -----------
            n_paths : int
                the number of paths to generate
            total_timesteps : int
                the total number of timesteps that the volatility process is generated over for
                a single day - the frequency of observations
            n_timesteps : int
                the number of timesteps to generate at this frequency
            seed : int
                random seed to be used

            Returns:
            -----------
            numpy array of simulated paths
        """
        self.seed = seed
        np.random.seed(self.seed)
        if n_timesteps is None:
            n_timesteps = total_timesteps
        vol_paths = self._generate_vol(self.vol, n_paths, total_timesteps, n_timesteps, **kwargs)
        self.vol_paths = vol_paths
        W = np.random.normal(loc=0.0, scale=np.sqrt(1/total_timesteps), size=(n_paths, n_timesteps))
        X = self.X_0 + np.cumsum(self.vol_paths * W, axis=-1)
        X = np.concatenate([np.tile(self.X_0, (n_paths, 1)), X], axis=-1)
        self.paths = X
        self.n_paths = n_paths
        self.n_timesteps = n_timesteps
        self.total_timesteps = total_timesteps
        return np.array(X, dtype='float32')

    def generate_from_vol(self, seed=None):
        """
            Generate the semi martingale from the volatility array

            Parameters:
            -----------
            seed: int
                random seed to be used for Brownian increments

            Returns:
            -----------
            numpy array of simulated paths:

        """
        self.seed = seed
        np.random.seed(seed)
        n_paths, n_timesteps = self.vol.shape
        self.vol_paths = self.vol
        W = np.random.normal(loc=0.0, scale=np.sqrt(1/n_timesteps), size=(n_paths, n_timesteps))
        X = self.X_0 + np.cumsum(self.vol * W, axis=-1)
        X = np.concatenate([np.tile(self.X_0, (n_paths, 1)), X], axis=-1)
        self.paths = X
        self.n_paths = n_paths
        self.n_timesteps = n_timesteps
        self.total_timesteps = n_timesteps
        return X

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
        int_var = np.sum(self.vol_paths**2, axis=-1)/self.total_timesteps
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
