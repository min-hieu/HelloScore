import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, t_channel):
        super().__init__()
        half_dim = t_channel // 2
        freq = np.log(10000) / (half_dim - 1)
        self.modulation = torch.exp(torch.arange(half_dim) * -freq)

    def forward(self, t):
        emb = t[:, None] * self.modulation[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_shapes):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_shapes[0]), nn.ReLU()]
        for i, hid_dim in enumerate(hid_shapes[1:]):
            layers += [nn.Linear(hid_shapes[i-1], hid_shapes[i]), nn.ReLU()]
        layers += [nn.Linear(hid_shapes[-1], out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
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
