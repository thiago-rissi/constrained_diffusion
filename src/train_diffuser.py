import torch
from diffusion.schedulers import *
from diffusion.samplers import *
from diffusion.trainers import *
from models.UNet import UNet, UNet2DWrapper, UNet_Tranformer, UNet_res
from utils.other_utils import *
import yaml
import sys
import pathlib

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    with open("config/train.yml", "r") as file:
        train_config = yaml.safe_load(file)

    with open("config/models.yml", "r") as file:
        models_config = yaml.safe_load(file)

    with open("config/datasets.yml", "r") as file:
        datasets_config = yaml.safe_load(file)

    dataset_name = train_config["dataset"]
    dataset_dict = datasets_config[dataset_name]
    img_size, img_channels, n_classes, batch_size = (
        dataset_dict["img_size"],
        dataset_dict["img_channels"],
        dataset_dict["n_classes"],
        train_config["batch_size"],
    )

    dataset = getattr(sys.modules[__name__], f"load_transformed_{dataset_name}")
    data, dataloader = dataset(
        img_size=img_size,
        img_channels=img_channels,
        n_classes=n_classes,
        batch_size=batch_size,
    )

    timesteps = 250
    sigma = 25.0
    eps = 1e-3

    # model = UNet2DWrapper()
    # model = UNet(
    #     timesteps,
    #     IMG_CH,
    #     IMG_SIZE,
    #     down_chs=(64, 64, 128),
    #     t_embed_dim=8,
    #     c_embed_dim=0,
    # )

    model = UNet_Tranformer(partial(marginal_prob_std, sigma=sigma, device=device))
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model.to(device)

    trainer = VESDETrainer(
        train_timesteps=None,
        sample_timesteps=torch.linspace(eps, 1.0, timesteps, device=device),
        device=device,
    )

    vesde_sampler = VESDESampler(
        device=device,
        img_ch=IMG_CH,
        img_size=IMG_SIZE,
    )

    num_epochs = 75
    losses, model = trainer.train(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        sampler=vesde_sampler,
        model_path="unet_transformer.pth",
        plot=False,
    )
