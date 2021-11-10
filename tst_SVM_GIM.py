#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: tst_SVM_GIM.py
@time: 2021/2/25 10:29
@desc:
'''
import argparse

import numpy as np

from bls.BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
from mycode.utils.evaluator import getAUC
from mycode.utils.data_process import *
# from mycode.utils.writeExcel import *


def OnehotEncoderInv(targets_matrix):
    Target = np.zeros((len(targets_matrix), 1))
    # targets = np.random.randint(0, 10, 1000)
    i = 0
    for x in range(len(targets_matrix)):
        # x = int(x)
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
    # bls(traindata, trainlabel, testdata, testlabel)
    # svm_test(traindata,trainlabel, testdata, testlabel)
    # mlp_test(traindata, trainlabel, testdata, testlabel)
    elm_test(traindata, trainlabel, testdata, testlabel)


def svm_test(traindata, trainlabel, testdata, testlabel):
    """
    SVM testing
    """
    from sklearn.svm import SVC
    from sklearn import metrics
    import xlwt

    clf = SVC(kernel='rbf')
    trainlabel = OnehotEncoderInv(trainlabel)
    testlabel = OnehotEncoderInv(testlabel)
    clf.fit(traindata, trainlabel)
    test_label_pred = clf.predict(testdata)

    tst_contain, tst_cm = output_metrics_2(testlabel, test_label_pred)
    print("Evaluation_metrics:")
    print(tst_contain)
    print("Confusion Matrix:")
    print(tst_cm)


def mlp_test(traindata, trainlabel, testdata, testlabel):
    from sklearn.neural_network import MLPClassifier
    from sklearn import metrics

    trainlabel = OnehotEncoderInv(trainlabel)
    testlabel = OnehotEncoderInv(testlabel)
    clf = MLPClassifier(solver='sgd', alpha=1e-5,
                        activation='relu',
                        # hidden_layer_sizes = (5, 2),
                        random_state = 1)
    clf.fit(traindata, trainlabel)
    test_label_pred = clf.predict(testdata)
    tst_contain, tst_cm = output_metrics_2(testlabel, test_label_pred)
    print("Evaluation_metrics:")
    print(tst_contain)
    print("Confusion Matrix:")
    print(tst_cm)

def elm_test(traindata, trainlabel, testdata, testlabel):
    from mycode.skelm import ELMClassifier
    from sklearn import metrics

    trainlabel = OnehotEncoderInv(trainlabel)
    testlabel = OnehotEncoderInv(testlabel)
    clf = ELMClassifier(n_neurons=100)
    clf.fit(traindata, trainlabel)
    test_label_pred = clf.predict(testdata)

    tst_contain, tst_cm = output_metrics_2(testlabel, test_label_pred)
    print("Evaluation_metrics:")
    print(tst_contain)
    print("Confusion Matrix:")
    print(tst_cm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attention_subnet')
    hparams = parser.parse_args()
    main(hparams)