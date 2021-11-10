#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: mydataset_tf.py
@time: 2021/2/20 20:03
@desc: For transfer Learning. Without splitting the dataset. Based on mydataset2.py
'''
import os
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import random_split
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import numpy as np
from torch.utils.data.dataset import Subset, ConcatDataset
import torch.utils.data as data

class build_mydl(LightningDataModule):
    def __init__(self,
                 train_PATH: str = None,
                 # test_PATH: str = None,
                 val_ratio: float = 0.1,
                 batch_size: int = 128,
                 num: int = 0,
                 num_workers: int = 0, #48
                 seed: int = 123,
                 *args,
                 **kwargs,
                 ):
        """
        Args:
            train_PATH: where to load training data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
        """
        super().__init__(*args, **kwargs)

        self.dims = (3, 224, 224)
        self.train_PATH = train_PATH
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.depth = num

    @property
    def num_classes(self):
        """
        Return:
            5
        """
        return 2

    def default_transforms(self):
        ds_transforms = transforms.Compose([
            transforms.Resize(size=(296, 296)),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.CenterCrop(size=(224, 224)),
            # transforms.Resize(size=(224,224)),
            # transforms.RandomResizedCrop(28, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]) # need change
        return ds_transforms

    def train_transform(self):
        train_transforms = self.default_transforms()
        return train_transforms

    def test_transforms(self):
        test_transform = transforms.Compose(
            [
                transforms.Resize(size=(224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) # need change
            ]
        )
        return test_transform

    def __getDS(self):
        """
                Get the train_ds, val_ds, test_ds.
        """
        transform = self.default_transforms() if self.train_transforms is None else self.train_transforms
        ds = ImageFolder(self.train_PATH,
                         transform=transform)
        length = len(ds)
        indices = list(range(length))
        split = int(np.floor(0.09091 * length))  # 分成11等分，8份train和2份val,1份test

        shuffle_dataset = True
        if shuffle_dataset:
            np.random.seed(123)
            np.random.shuffle(indices)
        indices_0 = indices[:2 * split]
        indices_1 = indices[2 * split:4 * split]
        indices_2 = indices[4 * split:6 * split]
        indices_3 = indices[6 * split:8 * split]
        indices_4 = indices[8 * split:10 * split]
        indices_test = indices[10 * split:]

        subset_0 = Subset(ds, indices_0)
        subset_1 = Subset(ds, indices_1)
        subset_2 = Subset(ds, indices_2)
        subset_3 = Subset(ds, indices_3)
        subset_4 = Subset(ds, indices_4)
        subset_test = Subset(ds, indices_test)

        ds_list = [subset_0, subset_1, subset_2, subset_3, subset_4]

        return subset_0, subset_1, subset_2, subset_3, subset_4, subset_test, ds_list

    def train_dl(self):
        """
        1.get the dataset from directory
        2.Dataloader
        """
        subset_0, subset_1, subset_2, subset_3, subset_4, subset_test, ds_list = self.__getDS()

        val_sub = subset_4
        subset = ConcatDataset(ds_list[0:4])

        dl_train = data.DataLoader(dataset=subset,
                                   batch_size=self.batch_size,
                                   shuffle=True)
        dl_val = data.DataLoader(dataset=val_sub,
                                 batch_size=self.batch_size,
                                 shuffle=False)
        dl_test = data.DataLoader(dataset=subset_test,
                                  batch_size=self.batch_size,
                                  shuffle=False
                                  )
        return dl_train, dl_val, dl_test
