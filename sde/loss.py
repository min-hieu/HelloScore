import torch
import torch.nn as nn

class DSMLoss():

    def __init__(self, alpha: float, diff_weight: bool):
        self.alpha       = alpha
        self.diff_weight = diff_weight
        self.mseloss     = nn.MSELoss()

    def __call__(self, t, x, model, target, diff_sq):
        pred = model(t, x)
        reg  = self.alpha * pred**2
        loss = self.mseloss(pred, target) + reg

        if self.diff_weight:
            loss = loss / diff_sq

        loss = loss.mean()
        return loss

class DDPMLoss():

    def __init__(self):
        # TODO
        return

    def __call__(self):
        # TODO
        return

class DDIMLoss():

    def __init__(self):
        # TODO
        return

    def __call__(self):
        # TODO
        return

class EDMLoss():

    def __init__(self):
        # TODO
        return

    def __call__(self):
        # TODO
        return
