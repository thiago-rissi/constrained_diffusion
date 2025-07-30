import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


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


def show_tensor_image(tensor: torch.Tensor):
    image = transform_tensor_to_image(tensor[0].detach().cpu())

    plt.imshow(image, cmap="gray")


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
