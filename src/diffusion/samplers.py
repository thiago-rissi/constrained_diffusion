import torch
from abc import abstractmethod, ABC
from pdlmc.pdlmc import PDLMCChain
from typing import Callable
from diffusion.schedulers import DDPMScheduler
import numpy as np
from models.UNet import marginal_prob_std, diffusion_coeff
from functools import partial
from typing import Any


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
    def sample(
        self,
        x_t: torch.Tensor,
        model: torch.nn.Module,
        timesteps: torch.Tensor,
        y: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred_t: torch.Tensor,
        *args,
        **kwargs,
    ) -> Any:
        pass


class VPSDESampler(Sampler):
    def __init__(
        self,
        scheduler: DDPMScheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
    ):
        super().__init__(device, img_ch, img_size)
        self.scheduler = scheduler

    def sample(
        self,
        x_t: torch.Tensor,
        model: torch.nn.Module,
        timesteps: torch.Tensor,
        y: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:

        dt = timesteps[1] - timesteps[0]

        batch_size = x_t.shape[0]
        for t in reversed(timesteps[1:]):
            t = torch.full((batch_size,), t.item(), device=self.device)
            noise_pred = model(x_t, t, y)
            x_t = self.reverse(x_t, t, dt, noise_pred)

        return x_t

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
        scheduler: DDPMScheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
        pdlmc_params: dict,
        marg_prob_params: dict,
        diff_coeff_params: dict,
        **kwargs,
    ):
        super().__init__(device, img_ch, img_size)
        self.marginal_prob_std = partial(
            marginal_prob_std, device=device, **marg_prob_params
        )
        self.diffusion_coeff = partial(
            diffusion_coeff, device=device, **diff_coeff_params
        )
        self.lmc_chain = PDLMCChain(**pdlmc_params)
        self.scheduler = scheduler

    def sample(
        self,
        x_t: torch.Tensor,
        model: torch.nn.Module,
        timesteps: torch.Tensor,
        y: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:

        dt = timesteps[1] - timesteps[0]

        batch_size = x_t.shape[0]
        for t in reversed(timesteps[1:]):
            t = torch.full((batch_size,), t.item(), device=self.device)
            noise_pred = model(x_t, t, y)
            x_t = self.reverse(x_t, t, dt, noise_pred)

        return x_t

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
        x_0 = x_t - pred_t
        sqrt_one_minus_alpha_bar_t = self.scheduler.sqrt_one_minus_alpha_bar[
            t, None, None, None
        ]
        beta_t = self.scheduler.betas[t, None, None, None]
        score = -pred_t / sqrt_one_minus_alpha_bar_t
        score = self.lmc_chain.run_chain(score=score, x=x_t, x_0=x_0)
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
        scheduler: DDPMScheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
    ):
        super().__init__(device, img_ch, img_size)
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        x_t: torch.Tensor,
        model: torch.nn.Module,
        timesteps: torch.Tensor,
        y: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:

        batch_size = x_t.shape[0]
        for t in reversed(timesteps):
            t = torch.full((batch_size,), t.item(), device=self.device)
            pred_noise = model(x_t, t, y)
            x_t = self.reverse(x_t, t, pred_noise)

        return x_t

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
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
        u_t = sqrt_alpha_inv_t * (x_t - pred_noise_coeff_t * pred_noise)
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
        update_steps: int,
        pdlmc_params: dict,
        marg_prob_params: dict,
        diff_coeff_params: dict,
        **kwargs,
    ):
        super().__init__(device, img_ch, img_size)
        self.marginal_prob_std = partial(
            marginal_prob_std, device=device, **marg_prob_params
        )
        self.diffusion_coeff = partial(
            diffusion_coeff, device=device, **diff_coeff_params
        )
        self.lmc_chain = PDLMCChain(**pdlmc_params)
        self.update_steps = update_steps

    def sample(
        self,
        x_t: torch.Tensor,
        model: torch.nn.Module,
        timesteps: torch.Tensor,
        y: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:

        dt = timesteps[1] - timesteps[0]
        sigma_T = self.marginal_prob_std(timesteps[-1])
        x_t = x_t * sigma_T[:, None, None, None]

        mean_x = torch.empty_like(x_t)
        batch_size = x_t.shape[0]
        for t in reversed(timesteps):
            t = torch.full((batch_size,), t.item(), device=self.device)
            score = model(x_t, t, y)
            x_t, mean_x = self.reverse(x_t, y, model, score, t, dt, self.update_steps)

        return mean_x

    def reverse_step(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        g: torch.Tensor,
        dt: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x_0 = x_t + (std**2)[:, None, None, None] * score
        score = self.lmc_chain.run_chain(score=score, x=x_t, x_0=x_0)
        mean_x = x_t + (g**2)[:, None, None, None] * score * dt
        x_t = mean_x + torch.sqrt(dt) * g[:, None, None, None] * torch.randn_like(x_t)

        return x_t, mean_x

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor | None,
        model: torch.nn.Module,
        score: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        update_steps: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        std = self.marginal_prob_std(t)
        g = self.diffusion_coeff(t)

        x_t, mean_x = self.reverse_step(x_t, score, g, dt, std)

        for _ in range(update_steps - 1):
            score = model(x_t, t, y)
            x_t, mean_x = self.reverse_step(x_t, score, g, dt, std)

        return x_t, mean_x


class VESDESampler(Sampler):
    def __init__(
        self,
        device: torch.device,
        img_ch: int,
        img_size: int,
        marg_prob_params: dict,
        diff_coeff_params: dict,
        **kwargs,
    ):
        super().__init__(device, img_ch, img_size)

        self.marginal_prob_std = partial(
            marginal_prob_std, device=device, **marg_prob_params
        )
        self.diffusion_coeff = partial(
            diffusion_coeff, device=device, **diff_coeff_params
        )

    @torch.no_grad()
    def sample(
        self,
        x_t: torch.Tensor,
        model: torch.nn.Module,
        timesteps: torch.Tensor,
        y: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:

        dt = timesteps[1] - timesteps[0]
        sigma_T = self.marginal_prob_std(timesteps[-1]).unsqueeze(0)
        x_t = x_t * sigma_T[:, None, None, None]

        mean_x = torch.empty_like(x_t)
        batch_size = x_t.shape[0]
        for t in reversed(timesteps):
            t = torch.full((batch_size,), t.item(), device=self.device)
            score = model(x_t, t, y)
            x_t, mean_x = self.reverse(x_t, score, t, dt)

        return mean_x

    def reverse(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        g = self.diffusion_coeff(t)

        mean_x = x_t + (g**2)[:, None, None, None] * score * dt
        x_t = mean_x + torch.sqrt(dt) * g[:, None, None, None] * torch.randn_like(x_t)

        return x_t, mean_x
