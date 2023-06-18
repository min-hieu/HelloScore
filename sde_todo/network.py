import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, t):
        '''
        input :
            - t [B,]
        output :
            - temb [B, out_dim]
        '''
        emb = None
        return emb

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_shapes):
        super().__init__()
        '''
        input :
            - in_dim []
            - out_dim []
            - hid_shapes [S]
        '''
        # TODO : implement simple MLP
        self.model = None

    def forward(self, x):
        '''
        input :
            - x [B,in_dim]
        output :
            - MLP(x) [B, out_dim]
        '''
        return self.model(x)

class Naive(nn.Module):

    def __init__(self, in_dim, enc_shapes, dec_shapes, z_dim):
        super().__init__()
        self.pe = PositionalEncoding(z_dim)
        self.x_enc = MLP(in_dim, z_dim, enc_shapes)
        self.t_enc = MLP(z_dim, z_dim, enc_shapes)
        self.dec = MLP(2*z_dim, in_dim, dec_shapes)

    def forward(self, t, x):
        temb = self.pe(t)
        temb = self.t_enc(temb)
        xemb = self.x_enc(x)
        h = torch.cat([xemb, temb], -1)

        return -self.dec(h)
