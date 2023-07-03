import abc
import torch
from tqdm import tqdm

class Sampler():

    def __init__(self, eps):
        self.eps = eps

    def get_sampling_fn(self, sde, dataset):

        def sampling_fn(N_samples):
            # Initial sample
            x = dataset[range(N_samples)]
            timesteps = torch.linspace(0, sde.T-self.eps, sde.N)

            x_hist = torch.zeros((sde.N, *x.shape))
            with torch.no_grad():
                for i, t in enumerate(tqdm(timesteps, desc='sampling')):
                    tt = torch.ones(x.shape[0]) * t
                    x = sde.predict_fn(tt, x)
                    x = sde.correct_fn(tt, x)
                    x_hist[i] = x

            out = x
            ntot = sde.N
            return out, ntot, timesteps, x_hist

        return sampling_fn
