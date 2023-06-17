import torch.nn as nn

class DSMLoss():

    def __init__(self, alpha: float, diff_weight: bool):
        self.alpha = alpha
        self.diff_weight = diff_weight
        self.mseloss = nn.MSELoss()

    def __call__(self, pred, target, diff_sq):
        reg = self.alpha * pred**2
        loss = self.mseloss(pred, target) + reg

        if self.diff_weight:
            loss = loss / diff_sq

        loss = loss.mean()
        return loss

class ISMLoss():

    def __init__(self):
        # TODO
        return

    def __call__(self):
        # TODO
        return

