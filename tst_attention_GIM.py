#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: tst_attention_GIM.py
@time: 2021/2/14 11:48
@desc: GIM, 2 classes
'''
import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm

from mycode.dataloaders.mydataset2 import build_mydl
from mycode.models.attention_resnet_images import attention_resnet_images
from mycode.utils.data_process import *


def main(hparams):
    # hparams.ds_name = ds_name
    print("Dataset_name:", hparams.ds_name)
    if hparams.ds_name == "chestmnist":
        task = "multi-label, binary-class"
    else:
        task = "train"

    depth_list = [6, 10, 14, 18, 19]
    # depth_list = [18]

    # Container for level-1
    Result_all = None
    Label_all = None
    Test_all = None
    Label_t_all = None

    for depth in depth_list:
        # Container for level-0
        Result = None
        Label = None
        Test = None
        Label_t = None

        for i in range(5):
            # DataLoaders
            dl = build_mydl(
                train_PATH=hparams.train_root,
                batch_size=hparams.batch_size,
                num=i,
            )
            train_dataloader, val_dataloader, test_dataloader = dl.train_dl()
            num_class = dl.num_classes

            path_str = 'tensorboard_logs_GIM/{dsname}/d{depth}'.format(dsname=hparams.ds_name, depth=depth * 10 + i)
            path_parent = Path(path_str)
            path_log = sorted(path_parent.glob('*/*.ckpt'))
            hparams.model = path_str + '/checkpoints/' + str(path_log[0].stem) + '.ckpt'

            # model
            model = attention_resnet_images.att_trained(num_classes=num_class,
                                                        depth=depth,
                                                        task=task,
                                                        path=hparams.model)
            model = model.eval().cuda(device=0)
            model.freeze()
            # output for BLS
            result, label = prepareDateBLS(num_class, val_dataloader, model)
            test, label_t = prepareDateBLS(num_class, test_dataloader, model)
            # Result = np.concatenate(Result,result)
            if Result is None:
                Result = result
                Label = label
                Test = test * 1. / 5
                Label_t = label_t

            else:
                Result = np.append(Result, result, axis=0)
                Label = np.append(Label, label, axis=0)
                Test = Test + test * 1. / 5
                Label_t = label_t

        if Result_all is None:
            Result_all = Result
            Label_all = Label
            Test_all = Test
            Label_t_all = Label_t

        else:
            Result_all = np.append(Result_all, Result, axis=1)
            Label_all = Label
            Test_all = np.append(Test_all, Test, axis=1)
            Label_t_all = Label_t

    # print("Data ok")
    traindata = Result_all
    trainlabel = Label_all
    testdata = Test_all
    testlabel = Label_t_all

    np.save("tensorboard_logs_images/{ds_name}/traindata.npy".format(ds_name=hparams.ds_name), traindata)
    np.save("tensorboard_logs_images/{ds_name}/trainlabel.npy".format(ds_name=hparams.ds_name), trainlabel)
    np.save("tensorboard_logs_images/{ds_name}/testdata.npy".format(ds_name=hparams.ds_name), testdata)
    np.save("tensorboard_logs_images/{ds_name}/testlabel.npy".format(ds_name=hparams.ds_name), testlabel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attention_subnet')
    parser.add_argument('--model', '-m', default='tensorboard_logs/att_subnet/d18/checkpoints/epoch=5.ckpt',
                        help='Model file')
    parser.add_argument('--train_root', type=str,
                        default="E:/Dataset/1.Medical_dataset/gim-yao_2class")
    # parser.add_argument('--train_root', type=str,
    #                     default="/home/fstpkw/ylData/gim-yao_2class")
    parser.add_argument('--ds_name', type=str, default='gim_2classes')
    parser.add_argument('--batch_size', type=int, default=128)
    hparams = parser.parse_args()
    main(hparams)
