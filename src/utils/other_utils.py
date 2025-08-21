from collections.abc import Callable
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from diffusion.samplers import Sampler


@torch.no_grad()
def sample_image(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    reverse: Callable,
    timesteps: torch.Tensor,
    **kwargs,
) -> torch.Tensor:

    dt = timesteps[1] - timesteps[0]
    for step in reversed(timesteps[1:]):
        t = step.float()
        t = torch.full((1,), t.item(), device=x_t.device)
        y = torch.randint(0, 10, (1,), device=x_t.device)
        pred_t = model(x_t, t, y)
        x_t = reverse(x_t, t, dt, pred_t, model=model, y=y, **kwargs)

    t = timesteps[0]
    t = torch.full((1,), t.item(), device=x_t.device)
    y = torch.randint(0, 10, (1,), device=x_t.device)
    pred_t = model(x_t, t, y)
    x_t = reverse(x_t, t, dt, pred_t, model=model, last=True, y=y, **kwargs)
    return x_t


def sample_images(
    model: torch.nn.Module,
    sampler: Sampler,
    timesteps: torch.Tensor,
    device: torch.device,
    sample_size: int,
    plot: bool,
    save: bool,
    save_path: str = "/usr/src/code/sampled_images.png",
    **kwargs,
) -> list:

    imgs = []
    for _ in tqdm(range(sample_size), desc="\t Sampling images"):
        x_t = torch.randn(
            (1, sampler.img_ch, sampler.img_size, sampler.img_size), device=device
        )
        y = torch.randint(0, 10, (1,), device=x_t.device)
        imgs.append(sampler.sample(x_t, model, timesteps, y, **kwargs))

    if plot:
        plt.figure(figsize=(4 * sample_size, 4))
        for i, img in enumerate(imgs):
            ax = plt.subplot(1, sample_size, i + 1)
            ax.axis("off")

            show_tensor_image(img)

        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig(save_path)

    return imgs


def save_animation(xs, gif_name, interval=300, repeat_delay=5000):
    fig = plt.figure()
    plt.axis("off")
    imgs = []

    for x_t in xs:
        im = plt.imshow(x_t, animated=True)
        imgs.append([im])

    animate = animation.ArtistAnimation(
        fig, imgs, interval=interval, repeat_delay=repeat_delay
    )
    animate.save(gif_name)


def transform_tensor_to_image(tensor: torch.Tensor):
    """
    Transforms a tensor into a PIL image.
    """
    reverse_transforms = transforms.Compose(
        [
            # transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
            transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
            transforms.ToPILImage(),
        ]
    )
    return reverse_transforms(tensor)


def show_tensor_image(tensor: torch.Tensor):
    image = transform_tensor_to_image(tensor[0].detach().cpu())
    plt.imshow(image)


def plot_generated_images(noise, result):
    plt.figure(figsize=(8, 8))
    nrows = 1
    ncols = 2
    samples = {"Noise": noise, "Generated Image": result}
    for i, (title, img) in enumerate(samples.items()):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_title(title)
        show_tensor_image(img)
    plt.show()


def calculate_class_proportions(
    imgs: list, classifier: torch.nn.Module, device: torch.device, n_classes: int
):
    imgs_tensor = torch.cat(imgs, dim=0)

    # Use the classifier to predict classes for each image
    with torch.no_grad():
        preds = classifier(imgs_tensor.to(device))
        pred_labels = preds.argmax(dim=1).cpu().numpy()

    # Calculate the proportion of each class
    class_counts = np.bincount(pred_labels, minlength=n_classes)
    class_proportions = class_counts / class_counts.sum()
    for i, prop in enumerate(class_proportions):
        print(f"{i}: {prop:.2f}", end="  ")


def calculate_mean_brightness(imgs: list) -> float:
    """
    Calculate the mean brightness of a list of images.
    """
    total_brightness = 0.0
    for img in imgs:
        total_brightness += img.mean().item()
    return total_brightness / len(imgs) if imgs else 0.0
