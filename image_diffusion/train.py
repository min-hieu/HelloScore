import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import AFHQDataModule, CIFAR100DataModule, CIFAR10DataModule, CelebADataModule, get_data_iterator, tensor_to_pil_image
from jutils import sysutil
from model import Diffusion, VarianceScheduler
from network import UNet
from pytorch_lightning import seed_everything
from scheduler import DDIMScheduler
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from dotmap import DotMap
import json

matplotlib.use("Agg")


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"
    config.num_diffusion_train_timesteps = 1000
    config.seed = 63

    now = sysutil.get_current_time()
    save_dir = Path(f"results/{config.dataset}-diffusion-{now}")
    save_dir.mkdir(exist_ok=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    if config.dataset == "cifar10":
        ds_module = CIFAR10DataModule("./data", batch_size=config.batch_size, num_workers=4)
        image_resolution = 32
    elif config.dataset == "cifar100":
        ds_module = CIFAR100DataModule("./data", batch_size=config.batch_size, num_workers=4)
        image_resolution = 128
    elif config.dataset == "celeba":
        ds_module = CelebADataModule("./data", batch_size=config.batch_size, num_workers=4)
        image_resolution = 64
    elif config.dataset == "afhq":
        ds_module = AFHQDataModule("./data", batch_size=config.batch_size, num_workers=4, max_num_images_per_cat=config.max_num_images_per_cat)
        image_resolution = 64
    else:
        raise ValueError(f"{config.dataset} is an invalid dataset name.")

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    var_scheduler = DDIMScheduler(
        config.num_diffusion_train_timesteps, beta_1=1e-4, beta_T=0.02, mode="linear"
    )
    var_scheduler.set_timesteps(20)
    # var_scheduler = VarianceScheduler(num_timesteps)

    net = UNet(
        T=config.num_diffusion_train_timesteps,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1, 
        use_cfg=args.use_cfg, 
        cfg_dropout=args.cfg_dropout, 
        num_classes=getattr(ds_module, "num_classes", None)
    )

    ddpm = Diffusion(net, var_scheduler)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )
    
    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                samples = ddpm.sample(4, return_traj=False, image_resolution=image_resolution)
                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")
                
                torch.save(ddpm.state_dict(), f"{save_dir}/last.ckpt")
                ddpm.train()

            img, label = next(train_it)
            img, label = img.to(config.device), label.to(config.device)
            loss = ddpm.get_loss(img, class_label=label)
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_num_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "celeba", "afhq"], default="cifar10")
    parser.add_argument("--max_num_images_per_cat", type=int, default=1000, help="max number of images per category for AFHQ dataset")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1, help="random dropout rate of making class label null")

    args = parser.parse_args()
    main(args)
