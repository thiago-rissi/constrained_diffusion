import torch
import matplotlib.pyplot as plt
from typing import Any
from torch.utils.data import DataLoader
from diffusion.samplers import Sampler
from diffusion.schedulers import DDPMScheduler
from tqdm import tqdm, trange
import torch.nn.functional as F
from utils.other_utils import sample_images
from abc import ABC, abstractmethod
from models.UNet import marginal_prob_std
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from typing import Any


class Trainer(ABC):
    def __init__(
        self,
        train_timesteps: None | torch.Tensor,
        sample_timesteps: torch.Tensor,
        device: torch.device,
    ):
        self.device = device
        self.train_timesteps = train_timesteps
        self.sample_timesteps = sample_timesteps

    @abstractmethod
    def get_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        **model_args,
    ) -> Any:
        pass

    @abstractmethod
    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def get_random_timestep(self, batch_size: int) -> torch.Tensor:
        pass

    def train(
        self,
        model: torch.nn.Module | Any,
        lr: float,
        dataloader: DataLoader,
        num_epochs: int,
        sampler: Sampler,
        model_path: str,
        sample_period: int,
        save_period: int,
        loss_path: str,
        plot: bool = True,
    ):
        losses = []
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98**epoch))
        for epoch in range(num_epochs):
            progress_bar = tqdm(
                range(len(dataloader)), desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            for step, (x_0, y) in enumerate(dataloader):
                loss = self.train_step(
                    model,
                    x_0,
                    y,
                    optimizer,
                    step,
                    sampler,
                    sample_period,
                    plot,
                )
                progress_bar.set_postfix(
                    Loss=loss.item(), MeanLoss=sum(losses[-100:]) / 100
                )
                progress_bar.update(1)
                losses.append(loss.item())

            if epoch % save_period:
                torch.save(model.state_dict(), model_path)

            scheduler.step()

        torch.save(model.state_dict(), model_path)
        self.plot_losses(torch.log10(torch.tensor(losses)), loss_path)
        return losses, model

    def train_step(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        step: int,
        sampler: Sampler,
        sample_period: int,
        plot: bool = True,
    ) -> torch.Tensor:
        """
        Perform a single training step.
        """
        optimizer.zero_grad()

        x_0 = x_0.to(self.device)
        y = y.to(self.device)
        t = self.get_random_timestep(x_0.size(0)).to(self.device)

        loss = self.get_loss(model, x_0, y, t)

        loss.backward()
        optimizer.step()

        if step % sample_period == 0 and plot:
            _ = sample_images(
                model,
                sampler,
                self.sample_timesteps,
                self.device,
                sample_size=1,
                plot=plot,
                save=True,
            )

        return loss

    def plot_losses(self, losses: torch.Tensor, loss_path: str) -> None:
        """
        Plot the training losses.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training log-loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(f"{loss_path}.png")
        plt.show()


class DDPMTrainer(Trainer):
    def __init__(
        self,
        scheduler: DDPMScheduler,
        train_timesteps: torch.Tensor,
        sample_timesteps: torch.Tensor,
        device: torch.device,
    ):
        self.scheduler = scheduler
        super().__init__(train_timesteps, sample_timesteps, device)

    def get_random_timestep(self, batch_size: int) -> torch.Tensor:
        """
        Get random timesteps for the batch.
        """
        return torch.randint(0, self.scheduler.T, (batch_size,), device=self.device)

    def get_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        **model_args,
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
        train_timesteps: torch.Tensor | None,
        sample_timesteps: torch.Tensor,
        device: torch.device,
        marg_prob_params: dict,
        eps: float,
        **kwargs,
    ):
        self.eps = eps
        self.marginal_prob_std = partial(
            marginal_prob_std, device=device, **marg_prob_params
        )
        super().__init__(train_timesteps, sample_timesteps, device)

    def get_random_timestep(self, batch_size: int) -> torch.Tensor:
        """
        Get random timesteps for the batch.
        """

        random_t = (
            torch.rand(batch_size, device=self.device) * (1.0 - self.eps) + self.eps
        )

        return random_t

    def get_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        **model_args,
    ) -> torch.Tensor:

        x_noisy, z, std = self.forward(x_0, t)
        score = model(x_noisy, t, y)
        return F.mse_loss(score * std[:, None, None, None], -z, reduction="mean")

    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> Any:

        z = torch.randn_like(x_0)
        std = self.marginal_prob_std(t)
        noise = z * std[:, None, None, None]
        x_t = x_0 + noise

        return x_t, z, std
