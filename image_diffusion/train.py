import argparse
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from image_diffusion.scheduler import DDIMScheduler
from network import UNet
from model import Diffusion, VarianceScheduler
from dataset import CIFAR10DataModule, tensor_to_pil_image, unnormalize
import matplotlib.pyplot as plt
from jutils import sysutil
from pathlib import Path
from pytorch_lightning import seed_everything

def main(args):
    """config"""
    device = f"cuda:{args.gpu}"
    num_timesteps = 1000
    epochs = 100
    now = sysutil.get_current_time()
    warmup_steps = 5000

    save_dir = Path(f"results/{now}")
    save_dir.mkdir(exist_ok=True)
    
    seed_everything(63)

    """######"""
    ds_module = CIFAR10DataModule("./data", batch_size=32, num_workers=4)
    train_dl = ds_module.train_dataloader()
    val_dl = ds_module.val_dataloader()
    
    var_scheduler = DDIMScheduler(num_timesteps, 
    # var_scheduler = VarianceScheduler(num_timesteps)

    net = UNet(
        T=num_timesteps, ch=128, ch_mult=[1,2,2,2], attn=[1],
        num_res_blocks=4, dropout=0.1)
   
    ddpm = Diffusion(net, var_scheduler)
    ddpm = ddpm.to(device)

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t+1) / warmup_steps, 1.0))

    losses = []
    for epoch in range(epochs):
        pbar = tqdm(train_dl)

        if epoch % 1 == 0:
            ddpm.eval()
            plt.plot(losses)
            plt.savefig(f"{save_dir}/loss.png")
            plt.close()
            
            # samples = ddpm.sample(4)
            samples = ddpm.ddim_sample(4, 50)
            for i in range(len(samples)):
                img = tensor_to_pil_image(samples[i])
                img.save(f"{save_dir}/epoch{epoch}_{i}.png")
           
            torch.save(ddpm.state_dict(), f"{save_dir}/last.ckpt")
        
        ddpm.train()
        for img, label in pbar:
            img = img.to(device)
            loss = ddpm.get_loss(img)
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
