import torch
from abc import abstractmethod, ABC
from pdlmc.pdlmc import PDLMCChain
from typing import Callable
from utils.schedulers import Scheduler
import numpy as np
from models.UNet import marginal_prob_std, diffusion_coeff
from functools import partial


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
        pred_t: torch.Tensor,
        *args,
        **kwargs,
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
        dt: torch.Tensor,
        pred_t: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:

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
        dt: torch.Tensor,
        pred_t: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:

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


class PDLMCVESampler(Sampler):
    def __init__(
        self,
        device: torch.device,
        img_ch: int,
        img_size: int,
        gfuncs: list[Callable],
        lmc_steps: int,
        step_size: float,
        step_size_lambda: float,
        sigma_init: float = 25.0,
    ):
        super().__init__(device, img_ch, img_size)
        self.marginal_prob_std = partial(
            marginal_prob_std, sigma=sigma_init, device=device
        )
        self.diffusion_coeff = partial(diffusion_coeff, sigma=sigma_init, device=device)
        self.sigma_init = sigma_init
        self.lmc_chain = PDLMCChain(
            gfuncs=gfuncs,
            step_size=step_size,
            step_size_lambda=step_size_lambda,
            lmc_steps=lmc_steps,
            device=device,
        )

    def reverse_step(
        self, x_t: torch.Tensor, score: torch.Tensor, g: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score = self.lmc_chain.run_chain(score=score, x=x_t)
        mean_x = x_t + (g**2)[:, None, None, None] * score * dt
        x_t = mean_x + torch.sqrt(dt) * g[:, None, None, None] * torch.randn_like(x_t)

        return x_t, mean_x

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        pred_t: torch.Tensor,
        y: torch.Tensor,
        model: torch.nn.Module,
        update_steps: int = 1,
        last: bool = False,
        **args,
    ) -> torch.Tensor:

        std = self.marginal_prob_std(t)
        if t.item() == 1:
            x_t = x_t * std[:, None, None, None]

        g = self.diffusion_coeff(t)

        score = pred_t
        x_t, mean_x = self.reverse_step(x_t, score, g, dt)

        for _ in range(update_steps - 1):
            score = model(x_t, t, y)
            x_t, mean_x = self.reverse_step(x_t, score, g, dt)

        if last:
            self.lmc_chain.reset()
            return mean_x

        return x_t


class VESDESampler(Sampler):
    def __init__(
        self,
        device: torch.device,
        img_ch: int,
        img_size: int,
        sigma_init: float = 25.0,
    ):
        super().__init__(device, img_ch, img_size)
        self.marginal_prob_std = partial(
            marginal_prob_std, sigma=sigma_init, device=device
        )
        self.diffusion_coeff = partial(diffusion_coeff, sigma=sigma_init, device=device)
        self.sigma_init = sigma_init

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        pred_t: torch.Tensor,
        last: bool = False,
        **kwargs,
    ) -> torch.Tensor:

        std = self.marginal_prob_std(t)
        if t.item() == 1:
            x_t = x_t * std[:, None, None, None]

        g = self.diffusion_coeff(t)

        score = pred_t
        mean_x = x_t + (g**2)[:, None, None, None] * score * dt

        if last:
            return mean_x

        x_t = mean_x + torch.sqrt(dt) * g[:, None, None, None] * torch.randn_like(x_t)

        return x_t
