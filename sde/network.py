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


class ResBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, in_dim)


class ResMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hid_shapes):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_shapes[0]), nn.ReLU()]
        for i, hid_dim in enumerate(hid_shapes[1:]):
            layers += [ResBlock(hid_shapes[i-1], hid_shapes[i])]
        layers += [nn.Linear(hid_shapes[-1], out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SimpleNet(nn.Module):

    def __init__(self, in_dim, enc_shapes, dec_shapes, z_dim):
        super().__init__()
        self.pe    = PositionalEncoding(z_dim)
        self.x_enc = MLP(in_dim, z_dim, enc_shapes)
        self.t_enc = MLP(z_dim, z_dim, enc_shapes)
        self.dec   = MLP(2*z_dim, in_dim, dec_shapes)

    def forward(self, t, x):
        temb = self.pe(t)
        temb = self.t_enc(temb)
        xemb = self.x_enc(x)
        h = torch.cat([xemb, temb], -1)

        return -self.dec(h)


class Block(nn.Module):
    def __init__(self, out_dim, groups=8):
        super().__init__()
        proj = nn.Conv2D(out_dim, kernel_shape=3, padding=(1, 1))
        norm = nn.GroupNorm(groups)
        act  = nn.silu
        self.model = nn.Sequential(proj, norm, act)

    def forward(self, x):
        return model(x)


class ResnetBlock(nn.Module):

    def __init__(self, out_dim, groups=8, change_dim=False):
        super().__init__()
        self.mlp    = nn.Sequential(nn.silu, nn.Linear(out_dim))
        self.block1 = Block(out_dim, groups=groups)
        self.block2 = Block(out_dim, groups=groups)
        self.res_conv = (
            nn.Conv2D(out_dim, kernel_shape=1, padding=(0, 0))
            if change_dim
            else lambda x: x
        )

    def forward(self, x, temb):
        h = self.block1(x)
        temb = self.mlp(temb)
        h = temb[:, None] + h
        h = self.block2(h)
        return h + self.res_conv(x)


def SpatialUpsample(dim):
    return nn.Conv2DTranspose(dim, kernel_shape=4, stride=2)


def SpatialDownsample(dim):
    return nn.Conv2D(dim, kernel_shape=4, stride=2, padding=(1, 1))


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1,),
        channels=1,
        resnet_block_groups=1,
    ):
        super().__init__()

        # determine dimensions
        init_dim = dim // 3 * 2
        self.init_conv = nn.Conv2D(init_dim, kernel_shape=7, padding=(3, 3))

        # time embeddings
        t_dim = dim * 4
        self.time_mlp = nn.Sequential(
                TimeEmbedding(dim),
                nn.Linear(t_dim),
                nn.gelu,
                nn.Linear(t_dim),
        )

        # layers
        self.downs = []
        dims = list(map(lambda m: dim * m, dim_mults))

        for ind, stage_dim in enumerate(dims):
            is_last = ind >= len(dims) - 1

            self.downs.append(
                [
                    ResnetBlock(
                        stage_dim, groups=resnet_block_groups, change_dim=True),
                    ResnetBlock(stage_dim, groups=resnet_block_groups),
                    # We don't apply spatial downsampling to the last stage. This is
                    # because we go from 28x28 -> 14x14 -> 7x7 in the 1st and 2nd
                    # stages and 7 can't be halved without a remainder, which
                    # would cause problems in the upsampling path.
                    SpatialDownsample(
                        stage_dim) if not is_last else lambda x: x,
                ]
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, groups=resnet_block_groups)
        self.mid_block2 = ResnetBlock(mid_dim, groups=resnet_block_groups)

        self.ups = []
        rev_dims = list(reversed(dims))
        for ind, stage_dim in enumerate(rev_dims):
            is_last = ind >= len(rev_dims) - 1

            self.ups.append(
                [
                    ResnetBlock(
                        stage_dim, groups=resnet_block_groups, change_dim=True),
                    ResnetBlock(stage_dim, groups=resnet_block_groups),
                    SpatialUpsample(stage_dim) if not is_last else lambda x: x,
                ]
            )

        self.final_block = ResnetBlock(dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2D(channels, kernel_shape=1, padding=(0, 0))

    def forward(self, time, x):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        h = []
        # downsample
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)


        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, upsample in self.ups:
            x = torch.concatenate((x, h.pop()), axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)

        x = self.final_block(x, t)
        return self.final_conv(x)
