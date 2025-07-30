import torch
import torch.nn.functional as F
from src.utils.other_utils import transform_tensor_to_image


def brightness_constraint(x: torch.Tensor, target: float = 0.4, tol: float = 0.05):
    """
    Ensures the mean brightness of the image is close to the target value.
    """
    x = torch.clamp(x, 0, 1)
    x = x.view(x.size(0), -1)
    mean_brightness = x.mean()
    g_val = F.relu(mean_brightness - target) - tol
    g_val = g_val.sum()

    return g_val


def center_constraint(
    x: torch.Tensor, max_radius: float = 5.0, center: tuple[int, int] = (10, 10)
):
    """
    Ensures digit center is within max_radius of target center.
    """
    cx_target, cy_target = center

    x_pos = torch.relu(x) + 1e-6
    B, C, H, W = x.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing="ij",
    )
    xs = xs.float().unsqueeze(0).unsqueeze(0)
    ys = ys.float().unsqueeze(0).unsqueeze(0)
    mass = x_pos.sum(dim=(2, 3), keepdim=True)
    cx = (xs * x_pos).sum(dim=(2, 3), keepdim=True) / mass
    cy = (ys * x_pos).sum(dim=(2, 3), keepdim=True) / mass
    dist = torch.sqrt((cx - cx_target) ** 2 + (cy - cy_target) ** 2)
    g_val = dist - max_radius
    return g_val.sum()


def classifier_constraint(
    x: torch.Tensor,
    classifier: torch.nn.Module,
    target_class: int = 3,
    epsilon: float = 0.2,
):
    """
    Creates a classification constraint function for constrained diffusion sampling.
    """

    classifier.eval()  # Make sure dropout/batchnorm aren't active

    logits = classifier(x)  # (batch, num_classes)
    ce_loss = F.cross_entropy(
        logits,
        torch.full((x.size(0),), target_class, device=x.device, dtype=torch.long),
        reduction="none",
    )
    g_val = ce_loss - epsilon  # (batch,)
    return g_val.sum()  # must return scalar for grad
