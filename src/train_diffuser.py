
import torch
from utils.ddpm_utils import *
from models import UNet
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

    # model = UNet.UNet2DWrapper()
    model = UNet.UNet(
        timesteps, IMG_CH, IMG_SIZE, down_chs=(64, 64, 128), t_embed_dim=8, c_embed_dim=0
    )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model.to(device)
    model = torch.compile(model)

    trainer = DDPMTrainer(scheduler=scheduler, device=device)
    ddpm_sampler = DDPMSampler(
        scheduler=scheduler,
        device=device,
        img_ch=IMG_CH,
        img_size=IMG_SIZE,
        ncols=ncols,
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
