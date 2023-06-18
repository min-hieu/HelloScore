import torch.nn as nn

class DSMLoss():

    def __init__(self, alpha: float, diff_weight: bool):
        '''
        input :
            - alpha []
            - diff_weight []
        '''
        # TODO

    def __call__(self, pred, target, diff_sq):
        '''
        input :
            - pred [B, D]
            - target [B, D]
        output : Loss
            - loss []

        '''
        # TODO : Implement the DSM Loss.
        loss = None
        return loss

class ISMLoss():

    def __init__(self):
        # TODO
        return

    def __call__(self):
        # TODO
        return

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
