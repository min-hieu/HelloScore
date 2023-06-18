import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets

def load_twodim(num_samples: int,
                dataset: str,
                dimension: int = 2):

    if dataset == 'gaussian_centered':
        sample = np.random.normal(size=(num_samples, dimension))
        sample = sample

    if dataset == 'gaussian_shift':
        sample = np.random.normal(size=(num_samples, dimension))
        sample = sample + 1.5

    if dataset == 'circle':
        X, y = datasets.make_circles(
            n_samples=num_samples, noise=0.0, random_state=None, factor=.5)
        sample = X * 4

    if dataset == 'scurve':
        X, y = datasets.make_s_curve(
            n_samples=num_samples, noise=0.0, random_state=None)
        init_sample = X[:, [0, 2]]
        scaling_factor = 2
        sample = (init_sample - init_sample.mean()) / \
            init_sample.std() * scaling_factor

    if dataset == 'swiss_roll':
        X, y = datasets.make_swiss_roll(
            n_samples=num_samples, noise=0.0, random_state=None, hole=True)
        init_sample = X[:, [0, 2]]
        scaling_factor = 2
        sample = (init_sample - init_sample.mean()) / \
            init_sample.std() * scaling_factor

    return torch.tensor(sample).float()


class TwoDimDataClass(Dataset):

    def __init__(self,
                 dataset_type: str,
                 N: int,
                 batch_size: int,
                 dimension = 2):

        self.X = load_twodim(N, dataset_type, dimension=dimension)
        self.name = dataset_type
        self.batch_size = batch_size
        self.dimension = 2

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

    def get_dataloader(self, shuffle=True):
        return DataLoader(self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
