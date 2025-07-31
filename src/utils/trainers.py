import torch
import matplotlib.pyplot as plt
from typing import Any
from torch.utils.data import DataLoader
from utils.samplers import Sampler
from utils.schedulers import Scheduler
from tqdm import tqdm
import torch.nn.functional as F
from utils.other_utils import sample_images
from abc import ABC, abstractmethod
import random
import numpy as np


class Trainer(ABC):
    def __init__(
        self,
        train_timesteps: torch.Tensor,
        sample_timesteps: torch.Tensor,
        device: torch.device,
    ):
        self.device = device
        self.train_timesteps = train_timesteps
        self.sample_timesteps = sample_timesteps

    @abstractmethod
    def get_loss(
        self, model: torch.nn.Module, x_0: torch.Tensor, t: torch.Tensor, **model_args
    ) -> Any:
        pass

    @abstractmethod
    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> Any:
        pass

    def train(
        self,
        model: torch.nn.Module | Any,
        dataloader: DataLoader,
        num_epochs: int,
        sampler: Sampler,
        plot: bool = True,
    ):
        losses = []
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(num_epochs):
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
                progress_bar.set_postfix(
                    Loss=loss.item(), MeanLoss=sum(losses[-100:]) / 100
                )
                progress_bar.update(1)
                losses.append(loss.item())

        if plot:
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
        t = self.train_timesteps[random.randint(0, len(self.train_timesteps) - 1)]
        t = torch.tensor([t] * x_0.size(0), device=self.device)

        loss = self.get_loss(model, x_0, t)

        loss.backward()
        optimizer.step()

        if step > 0 and step % 100 == 0 and plot:
            _ = sample_images(
                model,
                sampler.reverse,
                self.sample_timesteps,
                x_0.size(1),
                x_0.size(2),
                self.device,
                sample_size=1,
                plot=False,
                save=True,
            )

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
        plt.savefig("training_loss.png")
        plt.show()


class DDPMTrainer(Trainer):
    def __init__(
        self,
        scheduler: Scheduler,
        train_timesteps: torch.Tensor,
        sample_timesteps: torch.Tensor,
        device: torch.device,
    ):
        self.scheduler = scheduler
        super().__init__(train_timesteps, sample_timesteps, device)

    def get_loss(
        self, model: torch.nn.Module, x_0: torch.Tensor, t: torch.Tensor, **model_args
    ) -> torch.Tensor:
        x_noisy, score = self.forward(x_0, t)
        pred = model(x_noisy, t, *model_args)
        return F.mse_loss(score, pred)

    def forward(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class VESDETrainer(Trainer):
    def __init__(
        self,
        train_timesteps: torch.Tensor,
        sample_timesteps: torch.Tensor,
        device: torch.device,
    ):
        self.sigma = 0.01
        super().__init__(train_timesteps, sample_timesteps, device)

    def get_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        **model_args,
    ) -> torch.Tensor:
        eps = 1e-5
        t = t * (1.0 - eps) + eps
        x_noisy, score, std = self.forward(x_0, t)
        pred = model(x_noisy, t, *model_args)
        return F.mse_loss(score, pred * std[:, None, None, None], reduction="mean")

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> Any:

        z = torch.randn_like(x_0)
        std = self.marginal_prob_std(t, self.sigma)
        noise = z * std[:, None, None, None]
        x_t = x_0 + noise

        return x_t, -z, std

    def marginal_prob_std(self, t: torch.Tensor, sigma_init: float) -> torch.Tensor:

        t = torch.tensor(t, device=self.device)
        return torch.sqrt((sigma_init ** (2 * t) - 1.0) / 2.0 / np.log(sigma_init))
