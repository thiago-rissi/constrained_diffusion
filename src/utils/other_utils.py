from collections.abc import Callable
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils.samplers import Sampler


@torch.no_grad()
def sample_image(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    reverse: Callable,
    timesteps: torch.Tensor,
    *model_args,
) -> torch.Tensor:

    for step in reversed(timesteps):
        t = step.float()
        t = torch.full((1,), t, device=x_t.device)
        pred_t = model(x_t, t)
        x_t = reverse(x_t, t, pred_t)

    return x_t


def sample_images(
    model: torch.nn.Module,
    reverse: Callable,
    timesteps: torch.Tensor,
    img_ch: int,
    img_size: int,
    device: torch.device,
    sample_size: int,
    plot: bool,
    save: bool,
    save_path: str = "/usr/src/code/sampled_images.png",
    *model_args,
) -> list:

    imgs = []
    for _ in tqdm(range(sample_size), desc="\t Sampling images"):
        x_t = torch.randn((1, img_ch, img_size, img_size), device=device)
        imgs.append(sample_image(model, x_t, reverse, timesteps, device, *model_args))

    if plot:
        fig, axs = plt.subplots(1, sample_size, figsize=(5, sample_size * 5))
        for i, img in enumerate(imgs):
            axi = axs[i] if sample_size > 1 else axs
            axi.axis("off")
            show_tensor_image(axi, img.detach().cpu())
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


def load_fashionMNIST(data_transform, train=True):
    return torchvision.datasets.FashionMNIST(
        "./data/",
        download=True,
        train=train,
        transform=data_transform,
    )


def load_transformed_fashionMNIST(img_size, batch_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)
    train_set = load_fashionMNIST(data_transform, train=True)
    test_set = load_fashionMNIST(data_transform, train=False)
    data = torch.utils.data.ConcatDataset([train_set, test_set])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return data, dataloader


def load_MNIST(data_transform, train=True):
    return torchvision.datasets.MNIST(
        "./data/",
        download=True,
        train=train,
        transform=data_transform,
    )


def load_transformed_MNIST(img_size, batch_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    data_transform = transforms.Compose(data_transforms)
    train_set = load_MNIST(data_transform, train=True)
    test_set = load_MNIST(data_transform, train=False)
    data = torch.utils.data.ConcatDataset([train_set, test_set])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return data, dataloader


def transform_tensor_to_image(tensor: torch.Tensor):
    """
    Transforms a tensor into a PIL image.
    """
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
            transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
            transforms.ToPILImage(),
        ]
    )
    return reverse_transforms(tensor)


def show_tensor_image(ax: plt.Axes, tensor: torch.Tensor):
    image = transform_tensor_to_image(tensor[0].detach().cpu())
    ax.imshow(image)


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
    imgs: list, classifier: torch.nn.Module, device: str, n_classes: int
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
