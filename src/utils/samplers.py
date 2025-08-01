from networkx import sigma
import torch
import matplotlib.pyplot as plt
import utils.other_utils as other_utils
from abc import abstractmethod, ABC
from pdlmc.pdlmc import PDLMCChain
from typing import Callable
from utils.schedulers import Scheduler
import numpy as np


class Sampler(ABC):
    def __init__(
        self,
        device: torch.device,
        img_ch: int,
        img_size: int,
    ):
        self.device = device
        self.img_ch = img_ch
        self.img_size = img_size

    @abstractmethod
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        pred_t: torch.Tensor,
        **args,
    ) -> torch.Tensor:
        pass


class VPSDESampler(Sampler):
    def __init__(
        self,
        scheduler: Scheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
    ):
        super().__init__(device, img_ch, img_size)
        self.scheduler = scheduler

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        pred_t: torch.Tensor,
        deterministic: bool = False,
        **args,
    ) -> torch.Tensor:

        dt = -1
        t = t.int()

        if t[0] == 0:
            return x_t

        sqrt_one_minus_alpha_bar_t = self.scheduler.sqrt_one_minus_alpha_bar[
            t, None, None, None
        ]
        beta_t = self.scheduler.betas[t, None, None, None]
        score = -pred_t / sqrt_one_minus_alpha_bar_t
        drift = -0.5 * beta_t * x_t - beta_t * score

        # Euler-Maruyama step
        if deterministic:
            x_t = x_t + drift * dt
        else:
            z = torch.randn_like(x_t)
            x_t = x_t + drift * dt + torch.sqrt(beta_t * abs(dt)) * z

        return x_t


class PDLMCVPSampler(Sampler):
    def __init__(
        self,
        scheduler: Scheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
        gfuncs: list[Callable],
        lmc_steps: int,
        step_size: float,
        step_size_lambda: float,
    ):
        super().__init__(device, img_ch, img_size)

        self.scheduler = scheduler

        self.lmc_chain = PDLMCChain(
            gfuncs=gfuncs,
            step_size=step_size,
            step_size_lambda=step_size_lambda,
            lmc_steps=lmc_steps,
            device=device,
        )

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        pred_t: torch.Tensor,
        deterministic: bool = False,
        **args,
    ) -> torch.Tensor:

        dt = -1
        t = t.int()

        if t[0] == 0:
            return x_t

        sqrt_one_minus_alpha_bar_t = self.scheduler.sqrt_one_minus_alpha_bar[
            t, None, None, None
        ]
        beta_t = self.scheduler.betas[t, None, None, None]
        score = -pred_t / sqrt_one_minus_alpha_bar_t
        score = self.lmc_chain.run_chain(score=score, x=x_t)
        drift = -0.5 * beta_t * x_t - beta_t * score

        # Euler-Maruyama step
        if deterministic:
            x_t = x_t + drift * dt
        else:
            z = torch.randn_like(x_t)
            x_t = x_t + drift * dt + torch.sqrt(beta_t * abs(dt)) * z

        return x_t


class DDPMSampler(Sampler):
    def __init__(
        self,
        scheduler: Scheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
    ):
        super().__init__(device, img_ch, img_size)
        self.scheduler = scheduler

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        pred_t: torch.Tensor,
        **args,
    ) -> torch.Tensor:
        """
        The reverse diffusion process
        Returns the an image with the noise from time t removed and time t-1 added.
        model: the model used to remove the noise
        x_t: the noisy image at time t
        t: timestep
        model_args: additional arguments to pass to the model
        """
        t = t.int()
        pred_noise_coeff_t = self.scheduler.pred_noise_coeff[t]
        sqrt_alpha_inv_t = self.scheduler.sqrt_alpha_inv[t]
        u_t = sqrt_alpha_inv_t * (x_t - pred_noise_coeff_t * pred_t)
        if t[0] == 0:  # All t values should be the same
            return u_t  # Reverse diffusion complete!
        else:
            B_t = self.scheduler.betas[t - 1]  # Apply noise from the previos timestep
            new_noise = torch.randn_like(x_t)
            return u_t + torch.sqrt(B_t) * new_noise


class VESDESampler(Sampler):
    def __init__(
        self,
        device: torch.device,
        img_ch: int,
        img_size: int,
        sigma_init: float = 0.01,
    ):
        super().__init__(device, img_ch, img_size)
        self.sigma_init = sigma_init

    def marginal_prob_std(self, t: torch.Tensor, sigma_init: float) -> torch.Tensor:
        return torch.sqrt((sigma_init ** (2 * t) - 1.0) / 2.0 / np.log(sigma_init))

    def diffusion_coeff(self, t: torch.Tensor, sigma: float) -> torch.Tensor:
        return torch.tensor(sigma**t, device=self.device)

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        pred_t: torch.Tensor,
        **args,
    ) -> torch.Tensor:

        std = self.marginal_prob_std(t, self.sigma_init)
        if t.item() == 1:
            x_t = x_t * std[:, None, None, None]

        g = self.diffusion_coeff(t, std.item())

        score = pred_t
        mean_x = x_t + (g**2)[:, None, None, None] * score * dt
        x_t = mean_x + np.sqrt(dt) * g[:, None, None, None] * torch.randn_like(x_t)

        return x_t
