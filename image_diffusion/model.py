from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceScheduler(nn.Module):
    def __init__(self, num_steps, beta_1=1e-4, beta_T=0.02, mode="linear"):
        super().__init__()
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == "quad":
            betas = torch.linspace(beta_1**0.5, beta_T**0.5, num_steps) ** 2
        elif mode == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * np.pi / 2
            alphas = torch.cos(alphas).pow(2)
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(self, batch_size) -> torch.LongTensor:
        ts = np.random.choice(np.arange(self.num_steps), batch_size)
        return torch.from_numpy(ts)
    

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)
        x_noisy, noise = self.var_scheduler.add_noise(x0, timestep)
        noise_pred = self.network(x_noisy, timestep=timestep, class_label=class_label)

        loss = F.mse_loss(noise_pred.flatten(), noise.flatten(), reduction="mean")
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(self, batch_size, return_traj=False, 
            class_label: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = 1.0):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            class_label = class_label.to(self.device)
            null_class_label = torch.zeros_like(class_label)
            class_label_input = torch.cat([null_class_label, class_label])

        traj = [x_T]
        for t in self.var_scheduler.timesteps:
            x_t = traj[-1]
            if do_classifier_free_guidance:
                x_t_input = torch.cat([x_t]*2)
                noise_pred = self.network(x_t_input, timestep=t.to(self.device), class_label=class_label_input)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.network(x_t, timestep=t.to(self.device))

            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())
            

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
                "network": self.network,
                "var_scheduler": self.var_scheduler,
                } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)

    # @torch.no_grad()
    # def sample(self, batch_size=4, return_traj=False):
    #     x_T = torch.randn([batch_size, 3, 32, 32]).to(self.device)
       
    #     traj = {self.var_scheduler.num_steps-1: x_T}
    #     for t in range(self.var_scheduler.num_steps - 1, -1, -1):
    #         z = torch.randn_like(x_T) if t > 0 else torch.zeros_like(x_T)
    #         alpha = self.var_scheduler.alphas[t]
    #         alpha_prod = self.var_scheduler.alphas_cumprod[t]
            
    #         sigma = torch.sqrt(self.var_scheduler.betas[t])

    #         c0 = 1 / torch.sqrt(alpha)
    #         c1 = (1 - alpha) / torch.sqrt(1 - alpha_prod)

    #         x_t = traj[t]

    #         # noise_pred = self.network(x_t, timestep=torch.tensor([t]).to(x_t))
    #         noise_pred = self.network(x_t, timestep=torch.tensor([t]).to(x_t))

    #         x_next = c0 * (x_t - c1 * noise_pred) + sigma * z
    #         traj[t - 1] = x_next.detach()
    #         traj[t] = traj[t].cpu()

    #         if not return_traj:
    #             del traj[t]
    #     if return_traj:
    #         return traj
    #     else:
    #         return traj[-1]
    
    # @torch.no_grad()
    # def ddim_sample(self, batch_size=4, num_steps=20, return_traj=False):
    #     x_T = torch.randn([batch_size, 3, 32, 32]).to(self.device)
        
    #     train_timesteps = self.var_scheduler.num_steps
    #     step_ratio = train_timesteps // num_steps
    #     timesteps = (np.arange(num_steps) * step_ratio).round()[::-1].astype(np.int64)
    #     timesteps += 1

    #     def step(model_output, timestep, sample):
    #         prev_timestep = timestep - train_timesteps // num_steps

    #         alpha_prod_t = self.var_scheduler.alphas_cumprod[timestep]
    #         alpha_prod_t_prev = self.var_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0).to(self.device)

    #         beta_prod_t = 1 - alpha_prod_t

    #         pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    #         pred_epsilon = model_output

    #         prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + torch.sqrt(1 - alpha_prod_t_prev) * pred_epsilon
    #         return prev_sample

    #     traj = {timesteps[0]: x_T}
    #     for t in timesteps:
    #         t_prev = t - train_timesteps // num_steps
    #         x_t = traj[t]
            
    #         noise_pred = self.network(x_t, timestep=torch.tensor([t]).to(self.device))

    #         x_t_prev = step(noise_pred, t, x_t)
    #         traj[t_prev] = x_t_prev.detach()
    #         traj[t] = traj[t].cpu()
        
    #     if return_traj:
    #         return traj
    #     else:
    #         return traj[timesteps[-1]]
