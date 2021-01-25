from expOU import ExpOU
from heston import Heston
from sabr import SABR



def model_selector(model_name, **kwargs):
    if model_name == "expou":
        theta, beta = kwargs.get('theta', 1.0), kwargs.get('beta', 0.2)
        return ExpOU(theta, beta).generate(n_paths, total_timesteps, n_timesteps, seed)
    elif model_name == "heston":
        pass
    elif model_name == "sabr":
        pass
    elif model_name == "rough-heston":
        pass
    else:
        raise ValueError:
            print("Model Not Available")

