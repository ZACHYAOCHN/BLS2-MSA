#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zach Yao
@license: None
@contact: yaoliangchn@qq.com
@software: Pycharm Community.
@file: tst_AUC_GIM.py
@time: 2021/2/14 10:47
@desc:
'''
import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm
import xlwt

from mycode.dataloaders.mydataset2 import build_mydl
from mycode.models.attention_resnet_images import attention_resnet_images
from mycode.utils.evaluator import getAUC, getACC
from mycode.utils.data_process import *
from sklearn import metrics


def main(hparams):
    ds_list = [
        "gim_2classes",
    ]
    # excel write
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('orginal')
    ds_num = -1
    for ds_name in ds_list:
        hparams.ds_name = ds_name
        # ds_num: pointer for saving to xls.
        ds_num += 1
        # write the ds name

        worksheet.write(ds_num * 5, 0, label=hparams.ds_name)
        worksheet.write(ds_num * 5 +2, 0, label='train')
        worksheet.write(ds_num * 5 + 3, 0, label='vali')
        worksheet.write(ds_num * 5 + 4, 0, label='test')

        if hparams.ds_name == "chestmnist":
            task = "multi-label, binary-class"
        else:
            task = "train"
        depth_list = [6, 10, 14, 18]
        # depth_list = [6]

        for depth in depth_list:
            if depth == 19:
                depth_xl = 22
                worksheet.write(ds_num * 5, get_row(22,0)-1, label='depth={}'.format(depth))
            else:
                depth_xl = depth
            worksheet.write(ds_num * 5, get_row2(depth_xl, 0) - 1, label='depth={}'.format(depth))

            for i in range(5):
                # write the split number
                worksheet.write(ds_num * 5, get_row2(depth_xl, i), label='split-{}'.format(i))
                # write acc and auc--title
                worksheet.write(ds_num * 5 + 1, get_row2(depth_xl, i), label='acc')
                worksheet.write(ds_num * 5 + 1, get_row2(depth_xl, i) + 1, label='auc')
                worksheet.write(ds_num * 5 + 1, get_row2(depth_xl, i) + 2, label='sensity')
                worksheet.write(ds_num * 5 + 1, get_row2(depth_xl, i) + 3, label='specifity')
                worksheet.write(ds_num * 5 + 1, get_row2(depth_xl, i) + 4, label='F1')
                worksheet.write(ds_num * 5 + 1, get_row2(depth_xl, i) + 5, label='Precision')

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

                result_train, label_train = prepareDataAUC(num_class, train_dataloader, model)
                result_val, label_val = prepareDataAUC(num_class, val_dataloader, model)
                result_test, label_test = prepareDataAUC(num_class, test_dataloader, model)


                # old
                # train_acc_1 = getACC(label_train, result_train, task)
                # train_auc_1 = getAUC(label_train, result_train, task)
                # vali_acc_1 = getACC(label_val, result_val, task)
                # vali_auc_1 = getAUC(label_val, result_val, task)
                # test_acc_1 = getACC(label_test, result_test, task)
                # test_auc_1 = getAUC(label_test, result_test, task)


                result_train = OnehotEncoderInv(result_train)
                result_val = OnehotEncoderInv(result_val)
                result_test = OnehotEncoderInv(result_test)

                train_acc, train_auc, train_f1, train_recall, train_precision,cm_train, train_specifity = output_metrics(
                    label_train,result_train)
                vali_acc, vali_auc, vali_f1, vali_recall, vali_precision, cm_vali, vali_specifity = output_metrics(
                    label_val, result_val)
                test_acc, test_auc, test_f1, test_recall, test_precision, cm_test, test_specifity = output_metrics(
                    label_test, result_test)

                worksheet.write(ds_num * 5 + 2, get_row2(depth_xl, i), train_acc)
                worksheet.write(ds_num * 5 + 2, get_row2(depth_xl, i) + 1, train_auc)
                worksheet.write(ds_num * 5 + 2, get_row2(depth_xl, i) + 2, train_recall)
                worksheet.write(ds_num * 5 + 2, get_row2(depth_xl, i) + 3, train_specifity)
                worksheet.write(ds_num * 5 + 2, get_row2(depth_xl, i) + 4, train_f1)
                worksheet.write(ds_num * 5 + 2, get_row2(depth_xl, i) + 5, train_precision)

                worksheet.write(ds_num * 5 + 3, get_row2(depth_xl, i), vali_acc)
                worksheet.write(ds_num * 5 + 3, get_row2(depth_xl, i) + 1, vali_auc)
                worksheet.write(ds_num * 5 + 3, get_row2(depth_xl, i) + 2, vali_recall)
                worksheet.write(ds_num * 5 + 3, get_row2(depth_xl, i) + 3, vali_specifity)
                worksheet.write(ds_num * 5 + 3, get_row2(depth_xl, i) + 4, vali_f1)
                worksheet.write(ds_num * 5 + 3, get_row2(depth_xl, i) + 5, vali_precision)

                worksheet.write(ds_num * 5 + 4, get_row2(depth_xl, i), test_acc)
                worksheet.write(ds_num * 5 + 4, get_row2(depth_xl, i) + 1, test_auc)
                worksheet.write(ds_num * 5 + 4, get_row2(depth_xl, i) + 2, test_recall)
                worksheet.write(ds_num * 5 + 4, get_row2(depth_xl, i) + 3, test_specifity)
                worksheet.write(ds_num * 5 + 4, get_row2(depth_xl, i) + 4, test_f1)
                worksheet.write(ds_num * 5 + 4, get_row2(depth_xl, i) + 5, test_precision)

                # # output of cm_train
                worksheet.write(ds_num * 5 + 5, get_row2(depth_xl, i), "cm_train")
                worksheet.write(ds_num * 5 + 6,get_row2(depth_xl, i), np.float(cm_train[0,0]))
                worksheet.write(ds_num * 5 + 6, get_row2(depth_xl, i) +1, np.float(cm_train[0, 1]))
                worksheet.write(ds_num * 5 + 7, get_row2(depth_xl, i) , np.float(cm_train[1, 0]))
                worksheet.write(ds_num * 5 + 7, get_row2(depth_xl, i) +1, np.float(cm_train[1, 1]))
                #
                # # output of cm_test
                worksheet.write(ds_num * 5 + 5, get_row2(depth_xl, i) + 2, "cm_vali")
                worksheet.write(ds_num * 5 + 6, get_row2(depth_xl, i) + 2, np.float(cm_vali[0, 0]))
                worksheet.write(ds_num * 5 + 6, get_row2(depth_xl, i) + 3, np.float(cm_vali[0, 1]))
                worksheet.write(ds_num * 5 + 7, get_row2(depth_xl, i) + 2, np.float(cm_vali[1, 0]))
                worksheet.write(ds_num * 5 + 7, get_row2(depth_xl, i) + 3, np.float(cm_vali[1, 1]))
                # # output of cm_test
                worksheet.write(ds_num * 5 + 5, get_row2(depth_xl, i) +4 , "cm_test")
                worksheet.write(ds_num * 5 + 6, get_row2(depth_xl, i) + 4 , np.float(cm_test[0, 0]))
                worksheet.write(ds_num * 5 + 6, get_row2(depth_xl, i) + 5, np.float(cm_test[0, 1]))
                worksheet.write(ds_num * 5 + 7, get_row2(depth_xl, i) + 4, np.float(cm_test[1, 0]))
                worksheet.write(ds_num * 5 + 7, get_row2(depth_xl, i) + 5, np.float(cm_test[1, 1]))

                print("Dataset_name:", hparams.ds_name,"depth:",depth, "--Split-",i)
                # print("traindata:")
                # print("      acc:", train_acc, "auc:", train_auc)
                # print("validata:")
                # print("      acc:", vali_acc, "auc:", vali_auc)
                # print("testdata:")
                # print("      acc:", test_acc, "auc:", test_auc)
    workbook.save('./tensorboard_logs_GIM/PAM_GIM_new_1.xls')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIM_ACC&AUC')
    parser.add_argument('--model', '-m', default='tensorboard_logs/att_subnet/d18/checkpoints/epoch=5.ckpt',
                        help='Model file')
    parser.add_argument('--train_root', type=str,
                        default="E:/Dataset/1.Medical_dataset/gim-yao_2class")
    # parser.add_argument('--train_root', type=str,
    #                     default="/home/fstpkw/ylData/gim-yao_2class")
    # parser.add_argument('--train_root', type=str,
    #                     default="/home/fstpkw/ylData/Pneumonia_UM/train")
    # parser.add_argument('--test_root', type=str,
    #                     default="/home/fstpkw/ylData/Pneumonia_UM/validation")
    parser.add_argument('--ds_name', type=str, default='Pneumonia')
    parser.add_argument('--batch_size', type=int, default=4)
    hparams = parser.parse_args()
    main(hparams)
