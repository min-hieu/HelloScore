import torch
import torch.nn as nn


def sample_rademacher_like(x):
    return torch.randint(low=0, high=2, size=x.shape).to(x) * 2 - 1


def sample_gaussian_like(x):
    return torch.randn_like(x)


def sample_e(noise_type, x):
    return {
        'gaussian': sample_gaussian_like,
        'rademacher': sample_rademacher_like,
    }.get(noise_type)(x)


def get_div_approx(y, x, noise_type):
    e = sample_e(noise_type, x)
    e_dydx = torch.autograd.grad(y, x, e, retain_graph=True, create_graph=True)[0]
    div_y = e_dydx * e
    return div_y

def get_div_exact(y, x):
    assert x.requires_grad

    div = torch.zeros(x.shape[0]).to(x)
    for i in range(x.shape[1]):
        H_i = torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0]
        div += H_i[:, i]
    return div

def get_div(y, x, method='approx', approx_dist='gaussian'):
    if method == 'approx':
        return get_div_approx(y,x,approx_dist)
    elif method == 'exact':
        return get_div_exact(y,x)
    else:
        raise Exception('undefined method')

class DSMLoss():

    def __init__(self, alpha: float, diff_weight: bool):
        self.alpha       = alpha
        self.diff_weight = diff_weight
        self.mseloss     = nn.MSELoss()

    def __call__(self, t, x, model, y, diff_sq):
        y_hat = model(t, x)
        reg   = self.alpha * y_hat**2
        loss  = self.mseloss(y_hat, y) + reg

        if self.diff_weight:
            loss = loss / diff_sq

        loss = loss.mean()
        return loss


class ISMLoss():

    def __init__(self):
        pass

    def __call__(self, t, x, model):
        x.requires_grad = True
        y_hat = model(t, x)
        div_y_hat = get_div(y_hat, x, 'exact')
        loss = 0.5 * torch.norm(y_hat)**2 + div_y_hat

        loss = loss.mean()
        return loss


class SBJLoss():

    def __init__(self):
        pass

    def __call__(self, t, xf, zf, zb_fn):
        zb = zb_fn(t, xf)
        div_gzb = get_div_approx(zb, xf, 'gaussian')
        loss = 0.5 * (zf+zb)**2 + div_gzb
        loss = (loss * sb.dt) / xf

        return loss


class SBALoss():

    def __init__(self):
        pass

    def __call__(self, t, xf, zf, zb_fn):
        pass


class EDMLoss():

    def __init__(self):
        # TODO
        return

    def __call__(self):
        # TODO
        return
