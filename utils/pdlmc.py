from typing import Callable
from torch._functorch.apis import grad, vmap
import torch
import torch.nn.functional as F


class PDLMCChain:
    def __init__(
        self,
        gfuncs: list[Callable],
        step_size: float,
        step_size_lambda: float,
        lmc_steps: int,
        device: torch.device,
    ):
        self.gfuncs = gfuncs
        self.grad_gfuncs = [grad(g) for g in gfuncs]
        self.step_size = step_size
        self.step_size_lambda = step_size_lambda
        self.lmc_steps = lmc_steps
        self.device = device
        self.lambdas = torch.zeros(len(self.gfuncs), device=device)

    def run_chain(self, score: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.lmc_steps):

            grad_g_vals = torch.stack(
                [grad_gfunc(x) for grad_gfunc in self.grad_gfuncs], dim=-1
            )

            score = score - self.step_size * torch.inner(self.lambdas, grad_g_vals)

            g_vals = torch.tensor(
                [g_func(x) for g_func in self.gfuncs], device=self.device
            )
            self.lambdas = F.relu(self.lambdas + self.step_size_lambda * g_vals)

        return score
