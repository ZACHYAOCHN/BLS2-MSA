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

from bls.BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
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
    print(dslist * 10)
    bls(traindata, trainlabel, testdata, testlabel)

def bls(traindata, trainlabel, testdata, testlabel):
    #
    N1 = 5  # # of nodes belong to each window
    N2 = 5  # # of windows -------Feature mapping layer
    N3 = 5 # # of enhancement nodes -----Enhance layer
    L = 20  # # of incremental steps
    M1 = 20  # # of adding enhance nodes
    s = 0.9 # 0.8  # shrink coefficient
    C = 2 ** -30  # Regularization coefficient

    print('-------------------BLS_BASE---------------------------')
    Basic_container, B_Cm = BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
    print('-------------------BLS_ENHANCE------------------------')
    E_container, E_Cm = BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
    print('-------------------BLS_FEATURE&ENHANCE----------------')
    M2 = 5  # # of adding feature mapping nodes
    M3 = 7  # # of adding enhance nodes
    F_container, F_Cm = BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)

    # write to excel
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Basic')
    excelWriting_Classfy(worksheet,Basic_container,B_Cm)
    worksheet2 = workbook.add_sheet('Enhance')
    excelWriting_Classfy(worksheet2, E_container, E_Cm)
    worksheet3 = workbook.add_sheet('Feature')
    excelWriting_Classfy(worksheet3, F_container, F_Cm)
    workbook.save('./tensorboard_logs_GIM/t4.xls')

    print('-------------------END---------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attention_subnet')
    hparams = parser.parse_args()
    main(hparams)
