import abc
import torch

class SDE(abc.ABC):
    def __init__(self, N: int, T: int):
        super().__init__()
        self.N = N         # number of time step
        self.T = T         # end time

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        pass

    @abc.abstractmethod
    def marginal_prob(self, t, x):
        pass

    @abc.abstractmethod
    def prior_sampling(self, t, x):
        pass

    def reverse(self, model):
        N = self.N
        T = self.T
        sde_coeff = self.sde_coeff

        def get_reverse_drift_fn(model_fn):
            def reverse_drift_fn(t, x):
                # TO FILL
                drift, diffusion = sde_coeff(self.T-t, x)
                score = model_fn(self.T-t, x)
                reverse_drift = - drift + score * (diffusion ** 2)
                return reverse_drift
            return reverse_drift_fn

        class RSDE(self.__class__):
            def __init__(self, model):
                self.N = N
                self.T = T
                self.model = model
                self.reverse_drift_fn = get_reverse_drift_fn(model)

            def sde_coeff(self, t, x):
                _, diffusion = sde_coeff(self.T-t, x)
                drift = self.reverse_drift_fn(t, x)
                return drift, diffusion

        return RSDE(model)

class OrnsteinUhlenbeck(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        drift = -0.5 * x
        diffusion = torch.ones(x.shape)
        return drift, diffusion

    def marginal_prob(self, t, x):
        mean = torch.exp(-0.5 * t).unsqueeze(1) * x
        std = torch.sqrt(1 - torch.exp(-t)).unsqueeze(1) * torch.ones_like(x)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(shape)

class VPSDE(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, shape):
        return torch.randn(shape)

class VESDE(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, shape):
        return torch.randn(shape)

class SchrodingerBridge(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, shape):
        return torch.randn(shape)
