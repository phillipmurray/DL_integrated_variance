import numpy as np
from src.vols.base import SemiMartingale

class RandomVol(SemiMartingale):

    def __init__(self, distribution):
        """Constant Volatility Model.
        distribution: numpy.random.distribution object
         """
        self.distribution = distribution
        
    def generate(self, n_paths, total_timesteps, n_timesteps=None, seed=None, reshape_for_rnn=None, **kwargs):
        self.seed=seed
        np.random.seed(seed)

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
        var[:,0] = self.distribution(size=n_paths,**kwargs)**2
   
        
        spot = np.zeros_like(var)
        spot[:,0] = x_0 

        for t in range(1, n_timesteps):
            dW= np.random.normal(size=n_paths, loc=0.0,  scale=np.sqrt(dt))
            sigma = self.distribution(size=n_paths,**kwargs)
            var[:,t] = sigma**2
            
            spot[:,t] = spot[:,t-1] + sigma*dW
        
        vol = np.sqrt(var)
        
        if reshape_for_rnn:
            spot = np.expand_dims(spot, -1)
            vol = np.expand_dims(vol, -1)

        self.paths = spot.astype('float32')
        self.vol_paths = vol.astype('float32')

        return self.paths, self.vol_paths