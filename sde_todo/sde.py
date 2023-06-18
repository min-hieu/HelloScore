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
                # TODO : implement the reverse drift function
                return reverse_drift
            return reverse_drift_fn

        class RSDE(self.__class__):
            def __init__(self, model):
                self.N = N
                self.T = T
                self.model = model

            def sde_coeff(self, t, x):
                # TODO
                return drift, diffusion

        return RSDE(model)

class OrnsteinUhlenbeck(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        # TODO : get the drift and diffusion coefficient of the forward process
        return drift, diffusion

    def marginal_prob(self, t, x):
        # TODO : derive and implement the marginal probability P(X_t|X_0)
        return mean, std

    def prior_sampling(self, shape):
        # TODO : sample from prior distribution (normal)
        return

class VESDE(SDE):
    def __init__(self, N=100, T=1, sigma_min=0.01, sigma_max=50):
        super().__init__(N, T)
        # TODO : setup sigmas

    def sde_coeff(self, t, x):
        # TODO : implement the forward diffusion coefficient
        return drift, diffusion

    def prior_sampling(self, shape):
        # TODO : prior sampling of VESDE
        return None


class VPSDE(SDE):
    def __init__(self, N=1000, T=1, beta_min=0.1, beta_max=20):
        super().__init__(N, T)
        # TODO : setup sqrt_alpha_bar

    def sde_coeff(self, t, x):
        # TODO : implement DDPM / VPSDE forward process
        return drift, diffusion

    def marginal_prob(self, x, t):
        # TODO : implement mean, std of P(X_t|X_0)
        return mean, std


class EDM(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, shape):
        raise NotImplementedError


class FBSDE(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, shape):
        raise NotImplementedError
