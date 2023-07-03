import abc
import torch
import numpy as np

class SDE(abc.ABC):
    def __init__(self, N: int, T: int):
        super().__init__()
        self.N = N         # number of time step
        self.T = T         # end time
        self.dt = T / N
        self.is_reverse = False
        self.is_bridge = False

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        return NotImplemented

    @abc.abstractmethod
    def marginal_prob(self, t, x):
        return NotImplemented

    @abc.abstractmethod
    def predict_fn(self, x):
        return NotImplemented

    @abc.abstractmethod
    def correct_fn(self, t, x):
        return NotImplemented

    def prior_sampling(self, x):
        return torch.randn_like(x)

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x) * (dt**0.5)

    def predict_fn(self, t, x, dt=None):
        dt = self.dt if dt is None else dt
        f, g = self.sde_coeff(t, x)
        x = x + f*self.dt + g*self.dw(x, dt)
        return x

    def correct_fn(self, t, x):
        return x

    def reverse(self, model):
        N = self.N
        T = self.T
        forward_sde_coeff = self.sde_coeff

        class RSDE(self.__class__):
            def __init__(self, score_fn):
                super().__init__(N, T)
                self.score_fn = score_fn
                self.is_reverse = True
                self.forward_sde_coeff = forward_sde_coeff

            def sde_coeff(self, t, x):
                f, g = self.forward_sde_coeff(self.T-t, x)
                s = self.score_fn(self.T-t, x)
                reverse_f = -f + s*(g**2)
                return reverse_f, g

            def ode_coeff(self, t, x):
                f, g = self.forward_sde_coeff(self.T-t, x)
                s = self.score_fn(self.T-t, x)
                reverse_f = -f + 0.5*s*(g**2)
                return reverse_f, 0

            def predict_fn(self, t, x, dt=None, ode=False):
                dt = self.dt if dt is None else dt
                f, g = self.ode_coeff(t, x) if ode else self.sde_coeff(t, x)
                x = x + f*self.dt + g*self.dw(x, dt)
                return x

        return RSDE(model)

class OU(SDE):
    def __init__(self, N=1000, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        f = -0.5 * x
        g = torch.ones(x.shape)
        return f, g

    def marginal_prob(self, t, x):
        log_mean_coeff = -0.5 * t
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1 - torch.exp((2 * log_mean_coeff)[:,None])) * torch.ones_like(x)
        return mean, std

class VESDE(SDE):
    def __init__(self, N=100, T=1, sigma_min=0.01, sigma_max=50):
        super().__init__(N, T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

    def sde_coeff(self, t, x):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        f = torch.zeros_like(x)
        log_mean_coeff = torch.tensor(np.log(self.sigma_max) - np.log(self.sigma_min))[:, None]
        g = sigma * torch.sqrt(2 * log_mean_coeff) * torch.ones_like(x)
        return f, g

    def marginal_prob(self, t, x):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t[:, None]
        mean = x
        return mean, std


class VPSDE(SDE):
    def __init__(self, N=1000, T=1, beta_min=0.1, beta_max=20):
        super().__init__(N, T)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def sde_coeff(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        f = -0.5 * beta_t[:, None] * x
        g = torch.sqrt(beta_t[:, None])
        return f, g

    def marginal_prob(self, t, x):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff[:, None])) * torch.ones_like(x)
        return mean, std


class EDM(SDE):
    def __init__(self, N=1000, T=1):
        super().__init__(N, T)


class SB(abc.ABC):
    def __init__(self, N=1000, T=1, forward_z=None, backward_z=None):
        super().__init__()
        self.N = N         # number of time step
        self.T = T         # end time
        self.dt = T / N
        self.is_reverse = False
        self.is_bridge  = True
        self.forward_z  = forward_z
        self.backward_z = backward_z

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        return NotImplemented

    def sb_coeff(self, t, x):
        assert(self.forward_z is not None)
        f, g = self.sde_coeff(t, x)
        sb_f = f + g*forward_z(t,x)
        return sb_f, g

    def predict_fn(self, t, x, dt=None):
        dt = self.dt if dt is None else dt
        f, g = self.sb_coeff(t, x)
        x = x + f*self.dt + g*self.dw(x, dt)
        return x

    def reverse(self, model):
        N = self.N
        T = self.T
        forward_z  = forward_z
        backward_z = backward_z
        forward_sde_coeff = self.sde_coeff

        class RSB(self.__class__):
            def __init__(self, model):
                super().__init__(N, T, forward_z, backward_z)
                self.is_reverse = True
                self.forward_sde_coeff = forward_sde_coeff

            def sb_coeff(self, t, x):
                return f, g

        return RSDE(model)

class OUSB(SB):
    def __init__(self, N=1000, T=1, forward_z=None, backward_z=None):
        super().__init__(N, T, forward_z, backward_z)

    def sde_coeff(self, t, x):
        f = -0.5 * x
        g = torch.ones(x.shape)
        return f, g
