import torch
from utils.schedulers import *
from utils.samplers import *
from utils.trainers import *
from models.UNet import UNet, UNet2DWrapper, UNet_Tranformer, UNet_res
from utils.other_utils import *

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    IMG_SIZE = 28
    IMG_CH = 3
    BATCH_SIZE = 128
    N_CLASSES = 10
    data, dataloader = load_transformed_CIFAR10(IMG_SIZE, BATCH_SIZE)

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

    num_epochs = 150
    losses, model = trainer.train(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        sampler=vesde_sampler,
        model_path="unet_transformer_cifar10.pth",
        plot=False,
    )
