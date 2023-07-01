import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_pil_image

def unnormalize(x0):
    # x: [C,H,W] or [B,C,H,W]
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(x0).view(3,1,1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).to(x0).view(3,1,1)

    x0 = x0 * std + mean
    return x0

def tensor_to_pil_image(x: torch.Tensor):
    """
    x: [C,H,W]
    """
    x = unnormalize(x)
    x = torch.clamp(x, 0, 1)
    return to_pil_image(x)


class CIFAR10DataModule(object):
    def __init__(self, root: str = "data", batch_size: int = 32, num_workers: int = 4):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.train_ds = CIFAR10(
            root=root, train=True, download=True, transform=self.transform
        )
        self.val_ds = CIFAR10(
            root=root, train=False, download=True, transform=self.transform
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
