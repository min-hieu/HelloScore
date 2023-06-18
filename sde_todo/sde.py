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

class VESDE(SDE):
    def __init__(self, N=100, T=1, sigma_min=0.01, sigma_max=50):
        super().__init__(N, T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

    def sde_coeff(self, t, x):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        delta_log_var = np.log(self.sigma_max) - np.log(self.sigma_min)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * delta_log_var))
        return drift, diffusion

    def prior_sampling(self, shape):
        return torch.randn(shape) * self.sigma_max


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
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


class EDM(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, shape):
        return torch.randn(shape)


class FBSDE(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, shape):
        return torch.randn(shape)
