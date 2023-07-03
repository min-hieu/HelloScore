import torch
from tqdm import tqdm
from itertools import repeat
import matplotlib.pyplot as plt

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    return model

def get_step_fn(loss_fn, optimizer, ema, sde, model):
    def step_fn(batch):
        # uniformly sample time step
        t = sde.T*torch.rand(batch.shape[0])

        # forward diffusion
        mean, std = sde.marginal_prob(t, batch)
        _, diffusion = sde.sde_coeff(t, batch)
        diff_sq = diffusion ** 2
        z = torch.randn(mean.shape)
        xt = mean + std * z

        # get loss
        target = - (z / std).float()
        loss = loss_fn(t, xt.float(), model1, target, diff_sq)

        # optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update()

        return loss.item()

    return step_fn


def get_sb_step_fn(model_f, model_b, ema_f, ema_b, opt_f, opt_b, loss_fn, sb):
    def step_fn_alter(batch, forward):

    def step_fn_joint(batch):
        opt_f.zero_grad()
        opt_b.zero_grad()

    if joint:
        return step_fn_joint
    else:
        return step_fn_alter


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def train_diffusion(dataloader, step_fn, N_steps, plot=False):
    pbar = tqdm(range(N_steps), bar_format="{desc}{bar}{r_bar}", mininterval=1)
    loader = iter(repeater(dataloader))

    log_freq = 200
    loss_history = torch.zeros(N_steps//log_freq)
    for i, step in enumerate(pbar):
        batch = next(loader)
        loss = step_fn(batch)

        if step % log_freq == 0:
            loss_history[i//log_freq] = loss
            pbar.set_description("Loss: {:.3f}".format(loss))

    if plot:
        plt.plot(range(len(loss_history)), loss_history)
        plt.show()
