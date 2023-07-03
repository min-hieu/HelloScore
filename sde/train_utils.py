import torch
from tqdm import tqdm
from itertools import repeat
import matplotlib.pyplot as plt 

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
        loss = loss_fn(t, xt.float(), model, target, diff_sq)

        # optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update()

        return loss.item()

    return step_fn


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
