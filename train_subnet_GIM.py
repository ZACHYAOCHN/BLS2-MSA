#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: train_subnet_GIM.py
@time: 2021/2/10 15:47
@desc: 2 classes, GIM
'''
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from mycode.dataloaders.mydataset2 import build_mydl
from mycode.models.attention_resnet_images import attention_resnet_images


def main(hparams):
    # seeding
    pl.seed_everything(hparams.seed)

    depth_list = [6, 10, 14, 18, 19]
    # depth_list = [18, 19]
    for depth in depth_list:
        # DataLoaders
        for i in range(5):
            dl = build_mydl(
                train_PATH=hparams.train_root,
                batch_size=hparams.batch_size,
                num=i,
            )
            train_dataloader,val_dataloader,test_dataloader = dl.train_dl()
            num_class = dl.num_classes
            print("Dataset_name:", hparams.ds_name)

            if hparams.ds_name == "chestmnist":
                task = "multi-label, binary-class"
            else:
                task = "train"
            # model
            model = attention_resnet_images.att_res18(num_classes=num_class, depth=depth, task=task)

            # Trainer
            logger = TensorBoardLogger(
                save_dir=os.path.join(os.getcwd(), "tensorboard_logs_temp210927"),
                version='d{num}'.format(num=depth * 10 + i),
                name='{ds}'.format(ds=hparams.ds_name),
                log_graph=True
            )

            trainer = pl.Trainer(gpus=hparams.gpus, min_epochs=10, max_epochs=hparams.epochs,
                                 progress_bar_refresh_rate=2,
                                 profiler=True, logger=logger, auto_lr_find=True,
                                 callbacks=[EarlyStopping(patience=10, monitor='val_accuracy')])
            trainer.fit(model, train_dataloader, val_dataloader)
            # Testing
            trainer.test(test_dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_root', type=str,
                        default="F:/Dataset/1.Medical_dataset/gim-yao_2class")
    parser.add_argument('--ds_name', type=str, default='gim-yao_2class')
    args = parser.parse_args()
    main(args)

