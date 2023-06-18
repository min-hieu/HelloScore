import abc
import torch
from tqdm import tqdm

class Predictor(abc.ABC):

    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    @abc.abstractmethod
    def update_fn(self, t, x):
        pass

class EulerMaruyamaPredictor(Predictor):

    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, t, x):
        dt = 1. / self.sde.N
        z = torch.randn(x.shape)
        drift, diffusion = self.sde.sde_coeff(t, x)
        x_mean = x + dt * drift
        x = x_mean + diffusion * (z * (dt ** 0.5))
        return x, x_mean

class Sampler():

    def __init__(self, eps):
        self.eps = eps

    def get_sampling_fn(self, sde, dataset):

        update_fn = EulerMaruyamaPredictor(sde).update_fn

        def sampling_fn(N_samples):
            # Initial sample
            x = dataset[range(N_samples)]
            timesteps = torch.linspace(0, sde.T-self.eps, sde.N)

            x_hist = torch.zeros((sde.N, *x.shape))
            with torch.no_grad():
                for i, t in enumerate(tqdm(timesteps, desc='sampling')):
                    t_vec = torch.ones(x.shape[0]) * t
                    x, _ = update_fn(t_vec, x)
                    x_hist[i] = x

            out = x
            ntot = sde.N
            return out, ntot, timesteps, x_hist

        return sampling_fn
