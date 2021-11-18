#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: tst_bls_images.py
@time: 2021/1/30 15:37
@desc: BLS classify
'''
import argparse

import numpy as np

from bls.AdaptiveBLS_s import AdaptiveBLS_s
import xlwt
from mycode.utils.writeExcel import *


def OnehotEncoderInv(targets_matrix):
    Target = np.zeros((len(targets_matrix), 1))
    i = 0
    for x in range(len(targets_matrix)):
        for i in range(len(targets_matrix[1])):
            if targets_matrix[x][i] == 1:
                Target[x] = i
    return Target


def main(hparams):
    """GIM"""
    dslist = "gim_2classes"
    traindata = np.load("tensorboard_logs_GIM/{ds}/traindata.npy".format(ds=dslist))
    trainlabel = np.load("tensorboard_logs_GIM/{ds}/trainlabel.npy".format(ds=dslist))
    testdata = np.load("tensorboard_logs_GIM/{ds}/testdata.npy".format(ds=dslist))
    testlabel = np.load("tensorboard_logs_GIM/{ds}/testlabel.npy".format(ds=dslist))
    validata = np.load("tensorboard_logs_GIM/{ds}/testdata.npy".format(ds=dslist))
    valilabel = np.load("tensorboard_logs_GIM/{ds}/testlabel.npy".format(ds=dslist))
    print(dslist * 10)
    L1 = 50
    L2 = 50
    # for train
    bls(traindata, trainlabel, validata, valilabel,L1,L2)
    # for test
    # bls(traindata, trainlabel, testdata, testlabel,L1,L2)
    
def bls(traindata, trainlabel, testdata, testlabel,L1, L2):
    #
    N1 = 5  # # of nodes belong to each window
    N2 = 5  # # of windows -------Feature mapping layer
    N3 = 5 # # of enhancement nodes -----Enhance layer
    M1 = 20  # # of adding enhance nodes
    s = 0.9 # 0.8  # shrink coefficient
    C = 2 ** -30  # Regularization coefficient

    train_container, train_Cm = AdaptiveBLS_s(traindata, trainlabel,  validata, valilabel, s, C, N1, N2, N3, L1, L2, M1, M2, M3)

    print('-------------------END---------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attention_subnet')
    hparams = parser.parse_args()
    main(hparams)
