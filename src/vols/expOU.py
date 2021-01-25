import numpy as np
from src.vols.base import SemiMartingale

class ExpOU(SemiMartingale):
    def __init__(self, theta=1.0, beta=0.2, rho=0.0):
        self.beta = beta
        self.theta = theta
        self.rho = rho

    def generate(self, n_paths, total_timesteps, n_timesteps=None, seed=None, **kwargs):
        self.seed=seed
        np.random.seed(seed)
        
        v_0 = kwargs.get('v_0', np.random.normal(size=n_paths, loc=0.0, scale=np.sqrt(1/(2*self.theta))))
        x_0 = kwargs.get('x_0', 1.0)
        reshape_for_rnn = kwargs.get('reshape_for_rnn', False)

        total_timesteps = int(total_timesteps)
        self.total_timesteps = total_timesteps
    
        if n_timesteps is None:
            n_timesteps = total_timesteps
        else:
            n_timesteps = int(n_timesteps)
        dt = 1/total_timesteps

        log_vol = np.zeros((n_paths, n_timesteps))
        log_vol[:,0] = v_0

        vol = np.exp(log_vol)

        spot = np.zeros_like(log_vol)
        spot[:,0] = x_0 

        for t in range(1, n_timesteps):
            dB = np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))
            dW = self.rho*dB + np.sqrt(1-self.rho**2)*np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))

            log_vol[:,t] = log_vol[:,t-1] * (1 - self.theta * dt) + self.beta*dB             
            vol[:,t] = np.exp(log_vol[:,t])

            spot[:,t] = spot[:,t-1] + vol[:,t-1]*dW
        
        if reshape_for_rnn:
            spot = np.expand_dims(spot, -1)
            vol = np.expand_dims(vol, -1)

        self.paths = spot.astype('float32')
        self.vol_paths = vol.astype('float32')

        return self.paths, self.vol_paths