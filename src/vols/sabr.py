import numpy as np
from src.vols.base import SemiMartingale


class SABR(SemiMartingale):

    def __init__(self, x_0=50.0, v_0=1.0, beta=0.5, alpha=1.0, rho=0.0):
        """Heston Model for Stochastic Volatility.
        Symbols as in Wikipedia page:
        https://en.wikipedia.org/wiki/Stochastic_volatility
        Assumptions:
        - Beta in [0,1],
        - Alpha >=0
        - Rho in (-1,1)
         """
        self.x_0 = x_0
        self.v_0 = v_0
        self.beta = beta
        self.alpha = alpha
        self.rho = rho

    def generate(self, n_paths, total_timesteps, n_timesteps=None, seed=None, **kwargs):
        self.seed=seed
        np.random.seed(seed)

        v_0 = kwargs.get('v_0', self.v_0)
        x_0 = kwargs.get('x_0', self.x_0)
        reshape_for_rnn = kwargs.get('reshape_for_rnn', False)

        total_timesteps = int(total_timesteps)
        self.total_timesteps = total_timesteps
    
        if n_timesteps is None:
            n_timesteps = total_timesteps
        else:
            n_timesteps = int(n_timesteps)
        dt = 1/total_timesteps
        
        vol = np.zeros((n_paths, n_timesteps))
        vol[:,0] = v_0
        
        spot = np.zeros_like(vol)
        spot[:,0] = x_0 
        for t in range(1, n_timesteps):
            dB = np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))
            dW = self.rho*dB + np.sqrt(1-self.rho**2)*np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))
            
            vol[:,t] = vol[:,t-1] + self.alpha*vol[:,t-1]*dB

            
            spot[:,t] = spot[:,t-1] + vol[:,t-1]*np.power(spot[:,t-1],self.beta)*dW
    
        if reshape_for_rnn:
            spot = np.expand_dims(spot, -1)
            vol = np.expand_dims(vol, -1)

        self.paths = spot.astype('float32')
        self.vol_paths = vol.astype('float32')

        return self.paths, self.vol_paths