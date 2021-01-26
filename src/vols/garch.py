import numpy as np
from src.vols.base import SemiMartingale


class GARCH(SemiMartingale):

    def __init__(self, theta=1.0, omega=0.04, xi=1.0, rho=0.0):
        """Heston Model for Stochastic Volatility.
        Symbols as in Wikipedia page:
        https://en.wikipedia.org/wiki/Stochastic_volatility
         """
        self.omega = omega
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def generate(self, n_paths, total_timesteps, n_timesteps=None, seed=None, **kwargs):
        self.seed=seed
        np.random.seed(seed)

        v_0 = kwargs.get('v_0', self.omega)
        x_0 = kwargs.get('x_0', 1.0)
        reshape_for_rnn = kwargs.get('reshape_for_rnn', False)

        total_timesteps = int(total_timesteps)
        self.total_timesteps = total_timesteps
    
        if n_timesteps is None:
            n_timesteps = total_timesteps
        else:
            n_timesteps = int(n_timesteps)
        dt = 1/total_timesteps
        
        var = np.zeros((n_paths, n_timesteps))
        var[:,0] = v_0
        
        spot = np.zeros_like(var)
        spot[:,0] = x_0 

        for t in range(1, n_timesteps):
            dB = np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))
            dW = self.rho*dB + np.sqrt(1-self.rho**2)*np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))
            
            var[:,t] = var[:,t-1] + self.theta*(self.omega - var[:,t-1])*dt + self.xi*var[:,t-1]*dB
            var[:,t] = np.abs(var[:,t])
            
            spot[:,t] = spot[:,t-1] + var[:,t-1]*dW
        
        vol = np.sqrt(var)
        
        if reshape_for_rnn:
            spot = np.expand_dims(spot, -1)
            vol = np.expand_dims(vol, -1)

        self.paths = spot.astype('float32')
        self.vol_paths = vol.astype('float32')

        return self.paths, self.vol_paths