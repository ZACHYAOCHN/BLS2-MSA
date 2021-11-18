#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time

from mycode.utils.evaluator import getAUC
from mycode.utils.data_process import *
from sklearn import metrics


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def prepareDataAUC(train_y, OutputOfTrain):
    len_ds = len(OutputOfTrain)
    num_class = len(OutputOfTrain[1])
    pred = np.zeros(shape=(len_ds, num_class))
    label = np.zeros(shape=(len_ds, 1))
    # targets = np.zeros(shape=(len_ds, 1))
    # j = 0
    for i in range(len_ds):
        # targets[j] = target[i]
        pred[i] = softmax(OutputOfTrain[i])
        label[i] = np.argmax(train_y[i])
        # j += 1
    return label, pred


def OnehotEncoder(classes, targets_arr):
    Target = np.zeros((len(targets_arr), classes))
    # targets = np.random.randint(0, 10, 1000)
    i = 0
    for i in range(len(targets_arr)):
        for j in range(classes):
            # x = int(x)
            if j == targets_arr[i]:
                Target[i][j] = 1

    return Target


def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count / len(Label), 5))


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))


def linear(data):
    return data


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk

def AdaptiveBLS_s(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L1, L2, M1,M2,M3, train_index):
    u = 0

    train_x = preprocessing.scale(train_x, axis=1)
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])

    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []

    Rslts_container = np.zeros([14, L + 1])
    Cm_contatiner = np.zeros([4, (L + 2) * 2])
    index = 0

    time_start = time.time()

    for i in range(N2):
        random.seed(i + u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T

        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.mean(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain, c)

    OutputWeight = pinvOfInput.dot(train_y)
    time_end = time.time()
    trainTime = time_end - time_start

    OutputOfTrain = np.dot(InputOfOutputLayerTrain, OutputWeight)
    OutputOfTrain_p = np.squeeze(np.asarray(OutputOfTrain))
    train_yp, OutputOfTrain_p = prepareDataAUC(train_y, OutputOfTrain_p)
    train_contain, train_cm = output_metrics(train_yp, OutputOfTrain_p)

    for i in range(6):
        Rslts_container[2 * i][0] = train_contain[i][0]
    for i in range(2):
        for j in range(2):
            Cm_contatiner[j][i] = train_cm[j][i]
    Rslts_container[2 * 6][0] = trainTime

    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (outputOfEachWindowTest - minOfEachWindow[i]) / \
                                                                  distOfMaxAndMin[i]
    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()
    testTime = time_end - time_start

    OutputOfTest_p = np.squeeze(np.asarray(OutputOfTest))
    test_yp, OutputOfTest_p = prepareDataAUC(test_y, OutputOfTest_p)

    tst_contain, tst_cm = output_metrics(test_yp, OutputOfTest_p)

    for i in range(6):
        Rslts_container[2 * i + 1][0] = tst_contain[i][0]
    Rslts_container[2 * 6 + 1][0] = testTime
    for i in range(2):
        for j in range(2):
            Cm_contatiner[j + 2][i] = tst_cm[j][i]

    WeightOfNewFeature2 = list()
    WeightOfNewFeature3 = list()

    for e in list(range(L1)):
        time_start = time.time()
        random.seed(e + N2 + u)

        weightOfNewMapping = 2 * random.random([train_x.shape[1] + 1, M1]) - 1
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)
        betaOfNewWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T

        Beta1OfEachWindow.append(betaOfNewWindow)
        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)
        distOfMaxAndMin.append(np.max(TempOfFeatureOutput, axis=0) - np.min(TempOfFeatureOutput, axis=0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput, axis=0))
        outputOfNewWindow = (TempOfFeatureOutput - minOfEachWindow[N2 + e]) / distOfMaxAndMin[N2 + e]

        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer, outputOfNewWindow])

        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0], 1))])
        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2 * random.random([M1 + 1, M2]) - 1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2 * random.random([M1 + 1, M2]).T - 1).T

        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)
        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)
        parameter1 = s / np.max(tempOfNewFeatureEhanceNodes)

        outputOfNewFeatureEhanceNodes = tansig(tempOfNewFeatureEhanceNodes * parameter1)

        if N2 * N1 + e * M1 >= M3:
            random.seed(67797325 + e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2 * N1 + (e + 1) * M1 + 1, M3) - 1)
        else:
            random.seed(67797325 + e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2 * N1 + (e + 1) * M1 + 1, M3).T - 1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)
        InputOfEnhanceLayerWithBias = np.hstack(
            [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s / np.max(tempOfNewEnhanceNodes)
        OutputOfNewEnhanceNodes = tansig(tempOfNewEnhanceNodes * parameter2)

        OutputOfTotalNewAddNodes = np.hstack(
            [outputOfNewWindow, outputOfNewFeatureEhanceNodes, OutputOfNewEnhanceNodes])

        tempOfInputOfLastLayes = np.hstack([InputOfOutputLayerTrain, OutputOfTotalNewAddNodes])

        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)

        if C.all() == 0:
            w = D.shape[1]
            B = (np.eye(w) - D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeight = pinvOfInput.dot(train_y)

        InputOfOutputLayerTrain = tempOfInputOfLastLayes
        time_end = time.time()
        Train_time = time_end - time_start
        Rslts_container[12][e + 1] = Train_time
        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        predictLabel_p = np.squeeze(np.asarray(predictLabel))
        train_yp, predictLabel_p = prepareDataAUC(train_y, predictLabel_p)
        train_contain, train_cm = output_metrics(train_yp, predictLabel_p)
        for i in range(6):
            Rslts_container[2 * i][e + 1] = train_contain[i][0]
        for i in range(2):
            for j in range(2):
                Cm_contatiner[j][(e + 1) * 2 + i] = train_cm[j][i]

        time_start = time.time()
        WeightOfNewMapping = Beta1OfEachWindow[N2 + e]

        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping)
        outputOfNewWindowTest = (outputOfNewWindowTest - minOfEachWindow[N2 + e]) / distOfMaxAndMin[N2 + e]
        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, outputOfNewWindowTest])
        InputOfEnhanceLayerWithBiasTest = np.hstack(
            [OutputOfFeatureMappingLayerTest, 0.1 * np.ones([OutputOfFeatureMappingLayerTest.shape[0], 1])])
        NewInputOfEnhanceLayerWithBiasTest = np.hstack(
            [outputOfNewWindowTest, 0.1 * np.ones([outputOfNewWindowTest.shape[0], 1])])
        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]
        OutputOfRelateEnhanceNodes = tansig(
            NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)
        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]
        OutputOfNewEnhanceNodes = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes) * parameter2)
        InputOfOutputLayerTest = np.hstack(
            [InputOfOutputLayerTest, outputOfNewWindowTest, OutputOfRelateEnhanceNodes, OutputOfNewEnhanceNodes])

        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)
        predictLabel_p = np.squeeze(np.asarray(predictLabel))
        test_yp, predictLabel_p = prepareDataAUC(test_y, predictLabel_p)

        time_end = time.time()
        Testing_time = time_end - time_start
        tst_contain, tst_cm = output_metrics(test_yp, OutputOfTest_p)

        index = e

        for i in range(6):
            Rslts_container[2 * i + 1][index + 1] = tst_contain[i][0]
        Rslts_container[2 * 6 + 1][index + 1] = Testing_time
        for i in range(2):
            for j in range(2):
                Cm_contatiner[j + 2][(index + 1) * 2 + i] = tst_cm[j][i]

        if train_index == 1:  # for train
            if Rslts_container[1][index + 1] <= Rslts_container[1][index]:
                print("Length:", index)
                break

    parameterOfShrinkAdd = []
    for e in list(range(L2)):
        time_start = time.time()
        if N1 * N2 >= M1:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M1) - 1)
        else:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M1).T - 1).T

        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s / np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd * parameterOfShrinkAdd[e])

        tempOfLastLayerInput = np.hstack([tempOfInputOfLastLayes, OutputOfEnhanceLayerAdd])

        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T, D)).I.dot(np.dot(D.T, pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        Rslts_container[12][e + 1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)

        OutputOfTrain1_p = np.squeeze(np.asarray(OutputOfTrain1))
        train_yp, OutputOfTrain1_p = prepareDataAUC(train_y, OutputOfTrain1_p)
        train_contain, train_cm = output_metrics(train_yp, OutputOfTrain1_p)

        for i in range(6):
            Rslts_container[2 * i][e+index + 1] = train_contain[i][0]
        for i in range(2):
            for j in range(2):
                Cm_contatiner[j][(e+index + 1) * 2 + i] = train_cm[j][i]

        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(
            InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)

        OutputOfTest_p = np.squeeze(np.asarray(OutputOfTest1))
        test_yp, OutputOfTest_p = prepareDataAUC(test_y, OutputOfTest_p)
        Test_time = time.time() - time_start

        tst_contain, tst_cm = output_metrics(test_yp, OutputOfTest_p)

        for i in range(6):
            Rslts_container[2 * i + 1][e+index + 1] = tst_contain[i][0]
        Rslts_container[2 * 6 + 1][e+index + 1] = Test_time
        for i in range(2):
            for j in range(2):
                Cm_contatiner[j + 2][(e+index + 1) * 2 + i] = tst_cm[j][i]

        if index == 1:  # for train
            if Rslts_container[1][e+index + 1] <= Rslts_container[1][e+index]:
                print("Length_enhance:", e)
                break

    return Rslts_container, Cm_contatiner
