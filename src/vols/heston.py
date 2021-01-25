import numpy as np


class Heston():
    def __init__(self, theta=1.0, omega=0.2, xi = 1):
        """Heston Model for Stochastic Volatility.
        Symbols as in Wikipedia page:
        https://en.wikipedia.org/wiki/Stochastic_volatility
         """
        self.omega = omega
        self.theta = theta
        self.xi = xi 

    def generate(self, n_paths, total_timesteps, n_timesteps=None, seed=None):
        self.seed=seed
        np.random.seed(seed)
        if n_timesteps is None:
            n_timesteps = total_timesteps
        dt = np.sqrt(1/total_timesteps)
        log_vol = np.zeros((n_paths, n_timesteps))
        log_vol[:,0] = np.random.normal(size=n_paths, loc=0.0, scale=np.sqrt(1/(2*self.theta)))
        for t in range(1, n_timesteps):
            log_vol[:,t] = log_vol[:,t-1] * (1 - self.theta * dt) + np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))
        return np.exp(log_vol)