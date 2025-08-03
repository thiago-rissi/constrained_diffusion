import torch
from typing import Protocol


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
        self.t = torch.arange(timesteps, dtype=torch.float32).to(device)
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
