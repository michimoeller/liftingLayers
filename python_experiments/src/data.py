"""Dataset Loaders."""
import os
import h5py
import numpy as np
import random
import h5py

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from pytorch_tools.data import DataLoader


class H5FDataset(Dataset):

    def __init__(self, mode='train'):
        super(H5FDataset, self).__init__()
        self._mode = mode
        if self._mode == 'train':
            h5f = h5py.File('data/denoising/train.h5', 'r')
        elif self._mode == 'val':
            h5f = h5py.File('data/denoising/val.h5', 'r')
        elif self._mode == 'test':
            h5f = h5py.File('data/denoising/test.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self._mode == 'train':
            h5f = h5py.File('data/denoising/train.h5', 'r')
        elif self._mode == 'val':
            h5f = h5py.File('data/denoising/val.h5', 'r')
        elif self._mode == 'test':
            h5f = h5py.File('data/denoising/test.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)


class ResidualData(Dataset):

    def __init__(self, transform=None, target_transform=None):
        super(ResidualData, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        data = self.images[index]
        target = self.images[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # residual
        # noise = noisy_img - img
        target = data - target
        return data, target

    def __len__(self):
        return len(self.images)


class BSD500Patches40(ResidualData):
    """BSD 500."""

    def __init__(self, *args, **kwargs):
        super(BSD500Patches40, self).__init__(*args, **kwargs)

        self.images = H5FDataset(mode='train')


class Set12(ResidualData):
    """BSD68."""

    def __init__(self, *args, **kwargs):
        super(Set12, self).__init__(*args, **kwargs)

        self.images = H5FDataset(mode='val')


class BSD68(ResidualData):
    """BSD68."""

    def __init__(self, *args, **kwargs):
        super(BSD68, self).__init__(*args, **kwargs)

        self.images = H5FDataset(mode='test')


class AddGaussNoise(object):

    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, img):
        noise = torch.FloatTensor(img.size()).normal_(mean=self.mean, std=self.stddev / 255.0)
        # noisy_img = torch.clamp(img + noise, 0.0, 1.0)
        noisy_img = img + noise
        return noisy_img


def init_data_loaders(data_cfg, torch_cfg, logger=None):
    """Initialize train and test loaders for given dataset."""
    batch_sizes = data_cfg['batch_sizes']
    loader_kwargs = data_cfg['loader_kwargs'].copy()
    shuffle = loader_kwargs.pop('shuffle')

    data_transform = transforms.Compose([
        # transforms.ToTensor(),
        AddGaussNoise(data_cfg['noise']['mean'], data_cfg['noise']['stddev']), ])
    target_transform = None #transforms.ToTensor()

    train_dataset = BSD500Patches40(transform=data_transform,
                                    target_transform=target_transform)
    val_dataset = Set12(transform=data_transform,
                        target_transform=target_transform)
    test_dataset = BSD68(transform=data_transform,
                         target_transform=target_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_sizes["train"],
                              shuffle=shuffle,
                              **loader_kwargs)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_sizes["val"],
                            shuffle=False,
                            **loader_kwargs)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_sizes["test"],
                             shuffle=False,
                             **loader_kwargs)

    if logger is not None:
        logger("NUM SAMPLES in TRAIN/VAL/TEST: "
               f"{train_loader.num_samples}/{val_loader.num_samples}/{test_loader.num_samples}")
        logger("NUM BATCHES in TRAIN/VAL/TEST: "
               f"{len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    return train_loader, val_loader, test_loader, train_dataset[0][0].size()
