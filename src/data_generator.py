import numpy as np
import tensorflow as tf
from src.stoch_vol import ExpOU

class SemiMartingale():
    def __init__(self, X_0, vol, **kwargs):
        self.X_0 = X_0
        self.vol = vol
        self.seed = None
        self.n_paths = None
        self.n_timesteps = None
        self.paths = None

    def _generate_vol(self, vol, n_paths, n_timesteps, seed=None, **kwargs):
        self.seed = seed
        np.random.seed(seed)
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
                return ExpOU(theta, beta).generate(n_paths, n_timesteps, seed)


    def generate(self, n_paths, n_timesteps, seed=None, **kwargs):
        self.seed = seed
        np.random.seed(self.seed)
        vol_paths = self._generate_vol(self.vol, n_paths, n_timesteps, **kwargs)
        self.vol_paths = vol_paths
        W = np.random.normal(loc=0.0, scale=np.sqrt(1/n_timesteps), size=(n_paths, n_timesteps))
        X = self.X_0 + np.cumsum(self.vol_paths * W, axis=-1)
        X = np.concatenate([np.tile(self.X_0, (n_paths, 1)), X], axis=-1)
        self.paths = X
        self.n_paths = n_paths
        self.n_timesteps = n_timesteps
        return X

    def generate_from_vol(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
        n_paths, n_timesteps = self.vol.shape
        self.vol_paths = self.vol
        W = np.random.normal(loc=0.0, scale=np.sqrt(1 / n_timesteps), size=(n_paths, n_timesteps))
        X = self.X_0 + np.cumsum(self.vol * W, axis=-1)
        X = np.concatenate([np.tile(self.X_0, (n_paths, 1)), X], axis=-1)
        self.paths = X
        self.n_paths = n_paths
        self.n_timesteps = n_timesteps
        return X

    def integrated_variance(self, return_mean=False):
        int_var = np.mean(self.vol_paths**2, axis=-1)
        if return_mean:
            int_var = np.mean(int_var, axis=0)
        return int_var

    def get_increments(self):
        if self.paths is not None:
            return np.diff(self.paths, axis=-1)

    def reshape_for_rnn(self):
        if self.paths is not None:
            return np.reshape(self.paths, (self.n_paths, self.n_timesteps+1, 1))

    def get_total_increments(self):
        if self.paths is not None:
            return self.paths[:,-1] - self.paths[:,0]

    def get_norm_increments(self):
        if self.paths is not None:
            increments = self.get_total_increments()
            integrated_var = self.integrated_variance()
            return increments / integrated_var

    def _to_tensor(self):
        return tf.constant(self.paths)