from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear"):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts


class DDPMScheduler(BaseScheduler):
    def __init__(
        self, num_train_timesteps, beta_1, beta_T, mode="linear", sigma_type="small"
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

        self.sigma_type = sigma_type
        if sigma_type == "small":
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor(1.0), self.alphas_cumprod[-1:]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            )
        elif sigma_type == "large":
            sigmas = self.betas

        self.register_buffer("sigmas", sigmas)

    def step(self, sample: torch.Tensor, timestep: int, noise_pred: torch.Tensor):
        alpha_t = self.alphas[timestep]
        beta_t = self.betas[timestep]
        alpha_prod_t = self.alphas_cumprod[timestep]
        sigma_t = self.sigmas[timestep]

        c0 = 1 / torch.sqrt(alpha_t)
        c1 = beta_t / torch.sqrt(1 - alpha_prod_t)

        z = torch.randn_like(sample) if timestep > 1 else 0
        sample_prev = c0 * (sample - c1 * noise_pred) + sigma_t * z

        return sample_prev

    def add_noise(
        self,
        original_sample,
        timesteps: torch.IntTensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            sample: [B,C,H,W]
            timesteps: [B]
            noise: [B,C,H,W]
        Output:
            x_noisy: [B,C,H,W]
            noise: [B,C,H,W]
        """
        device = original_sample.device

        alphas_cumprod = self.alphas_cumprod.to(device)
        timesteps = timesteps.to(device)
        if noise is None:
            noise = torch.randn_like(original_sample)

        alpha_prod = alphas_cumprod[timesteps]
        alpha_prod = alpha_prod.flatten()
        while len(alpha_prod.shape) < len(original_sample.shape):
            alpha_prod = alpha_prod.unsqueeze(-1)

        sqrt_alpha_prod = torch.sqrt(alpha_prod)
        one_minus_sqrt_alpha_prod = torch.sqrt(1 - alpha_prod)

        noisy_sample = (
            sqrt_alpha_prod * original_sample + one_minus_sqrt_alpha_prod * noise
        )

        return noisy_sample, noise


class DDIMScheduler(BaseScheduler):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear"):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)

        one = torch.tensor(1.0)
        self.register_buffer("alpha_prod_0", one)

    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        if num_inference_timesteps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_timesteps ({num_inference_timesteps}) cannot exceed self.num_train_timesteps ({self.num_train_timesteps})"
            )

        self.num_inference_timesteps = num_inference_timesteps

        step_ratio = self.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def step(
        self,
        sample: torch.Tensor,
        timestep: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
    ):
        timestep_prev = (
            timestep - self.num_train_timesteps // self.num_inference_timesteps
        )
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[timestep_prev]
            if timestep_prev >= 0
            else self.alpha_prod_0
        )

        sigma_t_square = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev) * eta)
        sigma_t = sigma_t_square ** (0.5)

        pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

        direction_to_sample = torch.sqrt(1 - alpha_prod_t_prev - sigma_t_square) * noise_pred

        z = torch.randn_like(sample)
        random_noise = sigma_t * z

        sample_prev = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample
            + direction_to_sample
            + random_noise
        )

        return sample_prev

    def add_noise(
        self,
        original_sample,
        timesteps: torch.IntTensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            sample: [B,C,H,W]
            timesteps: [B]
            noise: [B,C,H,W]
        Output:
            x_noisy: [B,C,H,W]
            noise: [B,C,H,W]
        """
        device = original_sample.device

        alphas_cumprod = self.alphas_cumprod.to(device)
        timesteps = timesteps.to(device)
        if noise is None:
            noise = torch.randn_like(original_sample)

        alpha_prod = alphas_cumprod[timesteps]
        alpha_prod = alpha_prod.flatten()
        while len(alpha_prod.shape) < len(original_sample.shape):
            alpha_prod = alpha_prod.unsqueeze(-1)

        sqrt_alpha_prod = torch.sqrt(alpha_prod)
        one_minus_sqrt_alpha_prod = torch.sqrt(1 - alpha_prod)

        noisy_sample = (
            sqrt_alpha_prod * original_sample + one_minus_sqrt_alpha_prod * noise
        )

        return noisy_sample, noise
