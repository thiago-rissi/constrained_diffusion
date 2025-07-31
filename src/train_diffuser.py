
import torch
from utils.schedulers import *
from utils.samplers import *
from utils.trainers import *
from models.UNet import UNet, UNet2DWrapper
from utils.other_utils import *

torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    IMG_SIZE = 28
    IMG_CH = 1
    BATCH_SIZE = 128
    N_CLASSES = 10
    data, dataloader = load_transformed_MNIST(IMG_SIZE, BATCH_SIZE)
    
    ncols = 5
    timesteps = 250
    scheduler = CosineScheduler(timesteps=timesteps, device=device)

    # model = UNet2DWrapper()
    model = UNet(
        timesteps, IMG_CH, IMG_SIZE, down_chs=(64, 64, 128), t_embed_dim=8, c_embed_dim=0
    )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model.to(device)
    model = torch.compile(model)

    trainer = DDPMTrainer(scheduler=scheduler, device=device, train_timesteps=scheduler.t, sample_timesteps=scheduler.t)
    ddpm_sampler = DDPMSampler(
        scheduler=scheduler,
        device=device,
        img_ch=IMG_CH,
        img_size=IMG_SIZE,
    )

    num_epochs = 5
    losses, model = trainer.train(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        sampler=ddpm_sampler,
        plot=True,
    )
    
    torch.save(model, 'model.pkl')
