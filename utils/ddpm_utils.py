import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Callable, Protocol
from utils import other_utils
from torch.utils.data import DataLoader
from abc import abstractmethod, ABC
from utils.pdlmc import PDLMCChain
from tqdm import tqdm
from IPython.display import clear_output


class Scheduler(Protocol):
    alpha_bar: torch.Tensor
    sqrt_alpha_bar: torch.Tensor
    sqrt_one_minus_alpha_bar: torch.Tensor
    T: int
    alphas: torch.Tensor
    sqrt_alpha_inv: torch.Tensor
    pred_noise_coeff: torch.Tensor
    betas: torch.Tensor


class CosineScheduler(Scheduler):
    def __init__(self, timesteps: int, device: torch.device, s: float = 0.008):
        self.s = torch.tensor(s).to(device)
        self.t = torch.arange(timesteps, dtype=torch.float32).to(device)
        self.T = timesteps

        # Forward diffusion variables
        self.alpha_bar = self.cosine_alpha_bar()
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        # Reverse diffusion variables
        self.alphas = torch.empty_like(self.alpha_bar, device=device)
        self.alphas[0] = 1.0
        self.alphas[1:] = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.sqrt_alpha_inv = torch.sqrt(1 / self.alphas)
        self.pred_noise_coeff = (1 - self.alphas) / torch.sqrt(1 - self.alpha_bar)
        self.pred_noise_coeff[0] = 0.0  # No noise at t=0
        self.betas = 1 - self.alphas

    def cosine_alpha_bar(self) -> torch.Tensor:

        f_t = torch.cos(((self.t / self.T) + self.s) / (1 + self.s) * torch.pi / 2) ** 2
        return f_t / f_t[0]


class LinearScheduler(Scheduler):
    def __init__(
        self,
        timesteps: int,
        device: torch.device,
        B_start: float = 0.0001,
        B_end: float = 0.02,
    ):
        self.T = timesteps
        self.betas = torch.linspace(B_start, B_end, timesteps).to(device)
        self.device = device

        # Forward diffusion variables
        self.alpha = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)  # Mean Coefficient
        self.sqrt_one_minus_alpha_bar = torch.sqrt(
            1 - self.alpha_bar
        )  # St. Dev. Coefficient

        # Reverse diffusion variables
        self.sqrt_alpha_inv = torch.sqrt(1 / self.alpha)
        self.pred_noise_coeff = (1 - self.alpha) / torch.sqrt(1 - self.alpha_bar)


class Sampler(ABC):
    def __init__(
        self,
        scheduler: Scheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
        ncols: int,
        **args,
    ):
        self.scheduler = scheduler
        self.device = device
        self.img_ch = img_ch
        self.img_size = img_size
        self.ncols = ncols

    @abstractmethod
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        e_t: torch.Tensor,
        **args,
    ) -> torch.Tensor:
        pass

    @torch.no_grad()
    def sample_step(
        self,
        x_t: torch.Tensor,
        model: torch.nn.Module,
        plot_number: int,
        step_number: int,
        hidden_rows: float,
        *model_args,
        axis_on: bool = False,
    ):
        """
        Perform a single sampling step.
        """
        t = torch.full((1,), step_number, device=self.device).float()
        e_t = model(x_t, t, *model_args)  # Predicted noise
        x_t = self.reverse(x_t, t, e_t)
        if step_number % hidden_rows == 0:
            ax = plt.subplot(1, self.ncols + 1, plot_number)
            if not axis_on:
                ax.axis("off")
            other_utils.show_tensor_image(x_t.detach().cpu())
            plot_number += 1

        return plot_number, x_t

    @torch.no_grad()
    def sample_images(
        self,
        model: torch.nn.Module,
        *model_args,
        dt: int = 1,
        axis_on: bool = False,
    ) -> torch.Tensor:
        # Noise to generate images from
        x_t = torch.randn(
            (1, self.img_ch, self.img_size, self.img_size), device=self.device
        )
        plt.figure(figsize=(8, 8))
        hidden_rows = self.scheduler.T / self.ncols
        plot_number = 1

        # Go from T to 0 removing and adding noise until t = 0
        for i in range(0, self.scheduler.T, dt)[::-1]:
            plot_number, x_t = self.sample_step(
                x_t=x_t,
                model=model,
                plot_number=plot_number,
                step_number=i,
                hidden_rows=hidden_rows,
                axis_on=axis_on,
                *model_args,
            )

        plt.show()
        return x_t


class DDPMSampler(Sampler):
    def __init__(
        self,
        scheduler: Scheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
        ncols: int,
    ):
        super().__init__(
            scheduler, device, img_ch=img_ch, img_size=img_size, ncols=ncols
        )

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        e_t: torch.Tensor,
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
        u_t = sqrt_alpha_inv_t * (x_t - pred_noise_coeff_t * e_t)
        if t[0] == 0:  # All t values should be the same
            return u_t  # Reverse diffusion complete!
        else:
            B_t = self.scheduler.betas[t - 1]  # Apply noise from the previos timestep
            new_noise = torch.randn_like(x_t)
            return u_t + torch.sqrt(B_t) * new_noise


class VPSDESampler(Sampler):
    def __init__(
        self,
        scheduler: Scheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
        ncols: int,
    ):
        super().__init__(
            scheduler, device, img_ch=img_ch, img_size=img_size, ncols=ncols
        )

    @torch.no_grad()
    def reverse(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        e_t: torch.Tensor,
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
        score = -e_t / sqrt_one_minus_alpha_bar_t
        drift = -0.5 * beta_t * x_t - beta_t * score

        # Euler-Maruyama step
        if deterministic:
            x_t = x_t + drift * dt
        else:
            z = torch.randn_like(x_t)
            x_t = x_t + drift * dt + torch.sqrt(beta_t * abs(dt)) * z

        return x_t


class PDLMCSampler(Sampler):
    def __init__(
        self,
        scheduler: Scheduler,
        device: torch.device,
        img_ch: int,
        img_size: int,
        ncols: int,
        gfuncs: list[Callable],
        lmc_steps: int,
        step_size: float,
        step_size_lambda: float,
    ):
        super().__init__(
            scheduler, device, img_ch=img_ch, img_size=img_size, ncols=ncols
        )
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
        e_t: torch.Tensor,
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
        score = -e_t / sqrt_one_minus_alpha_bar_t
        score = self.lmc_chain.run_chain(score=score, x=x_t)
        drift = -0.5 * beta_t * x_t - beta_t * score

        # Euler-Maruyama step
        if deterministic:
            x_t = x_t + drift * dt
        else:
            z = torch.randn_like(x_t)
            x_t = x_t + drift * dt + torch.sqrt(beta_t * abs(dt)) * z

        return x_t


class DDPMTrainer:
    def __init__(self, scheduler: Scheduler, device: torch.device):
        self.scheduler = scheduler
        self.device = device

    def q(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        The forward diffusion process
        Returns the noise applied to an image at timestep t
        x_0: the original image
        t: timestep
        """
        t = t.int()
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = self.scheduler.sqrt_alpha_bar[t, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.scheduler.sqrt_one_minus_alpha_bar[
            t, None, None, None
        ]

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def get_loss(
        self, model: torch.nn.Module, x_0: torch.Tensor, t: torch.Tensor, *model_args
    ) -> torch.Tensor:
        x_noisy, noise = self.q(x_0, t)
        noise_pred = model(x_noisy, t, *model_args)
        return F.mse_loss(noise, noise_pred)

    def train(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        num_epochs: int,
        sampler: Sampler,
        plot: bool = True,
    ):
        losses = []
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            clear_output(wait=True)
            progress_bar = tqdm(
                range(len(dataloader)), desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            for step, (x_0, _) in enumerate(dataloader):
                loss = self.train_step(
                    model,
                    x_0,
                    optimizer,
                    losses,
                    epoch,
                    step,
                    sampler,
                    plot,
                )
                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(1)
                losses.append(loss.item())

        self.plot_losses(torch.log10(torch.tensor(losses)))
        return losses, model

    def train_step(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        losses: list[float],
        epoch: int,
        step: int,
        sampler: Sampler,
        plot: bool = True,
    ) -> torch.Tensor:
        """
        Perform a single training step.
        """
        optimizer.zero_grad()

        x_0 = x_0.to(self.device)
        t = torch.randint(0, self.scheduler.T, (x_0.size(0),), device=self.device)

        loss = self.get_loss(model, x_0, t)

        loss.backward()
        optimizer.step()

        if step > 0 and step % 100 == 0:
            print(
                f"Epoch {epoch+1}, Step: {step}, Mean loss: {sum(losses[-100:]) / 100}"
            )
            if plot:
                _ = sampler.sample_images(model)

        return loss

    def plot_losses(self, losses: torch.Tensor):
        """
        Plot the training losses.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training log-loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.show()
