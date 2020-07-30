import numpy as np

class ExpOU():
    def __init__(self, theta=1.0, beta=0.2):
        self.beta = beta
        self.theta = theta

    def generate(self, n_paths, n_timesteps, seed=None):
        self.seed=seed
        np.random.seed(seed)
        dt = np.sqrt(1/n_timesteps)
        log_vol = np.zeros((n_paths, n_timesteps))
        log_vol[:,0] = np.random.normal(size=n_paths, loc=0.0, scale=np.sqrt(1/(2*self.theta)))
        for t in range(1, n_timesteps):
            log_vol[:,t] = log_vol[:,t-1] * (1 - self.theta * dt) + np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))
        return np.exp(log_vol)