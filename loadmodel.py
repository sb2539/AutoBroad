import datetime
import os
import random
import torch
import torch.nn as nn
from torchsummary import summary
import pdb

import matplotlib.pyplot as plot
import copy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
import pylab as plt
import pandas as pd
from pyts.datasets import fetch_ucr_dataset, ucr_dataset_list
import time
import warnings
import numpy as np

from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning

from imgcoding.coding import Image_coding
from models.broadcycle import CycleNet, CycleMLP
from utils.utils import (
    count_parameters,
    min_max,
    z_score,
    stacking
)
from utils.customdata import CustomDataset
import argparse
# load model weight using argument parser
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-ucr', type=str, help="ucr data name")
    parser.add_argument('-channel', type=int)
    parser.add_argument('-class_num', type=int)
    parser.add_argument('-input_size', type=int)
    parser.add_argument('-num_test', type=int)
    parser.add_argument('-len_test', type=int)
    parser.add_argument('-imgcode', type=str, nargs='*')
    parser.add_argument('-stage', type=int)
    parser.add_argument('-cell', type=int)
    parser.add_argument('-dim', type=int)

    args = parser.parse_args()
    PATH = "/home/sin/PycharmProjects/please/save/"
    RPATH = PATH + args.ucr + "/"
    print(PATH)
    print(RPATH)
    data_name = args.ucr
    ucr = fetch_ucr_dataset(data_name, use_cache=False, data_home=None, return_X_y=False)
    ucr_test = ucr.data_test
    ucr_target_test = ucr.target_test
    ucr_test_re = resize(ucr_test, ((args.num_test, args.len_test, 1)))
    teencoder = LabelEncoder()
    teencoder.fit(ucr_target_test)
    ucr_target_test = teencoder.transform((ucr_target_test))

    class_num = args.class_num
    input_size = args.input_size
    stage_num = args.stage
    cell = args.cell
    start_dim = args.dim
    test_size = len(ucr_test_re)

    tran = Image_coding(decompose=True)
    test_rp, test_rerp, test_trrp, test_gas, test_regas, test_trgas, test_gad, test_regad, test_trgad, test_mk, test_remk, test_trmk, test_ctwav = tran(
        ucr_test_re)

    test_rp = resize(test_rp, ((test_size, input_size, input_size)))
    test_trrp = resize(test_trrp, ((test_size, input_size, input_size)))
    test_rerp = resize(test_rerp, ((test_size, input_size, input_size)))

    test_gad = resize(test_gad, ((test_size, input_size, input_size)))
    test_trgad = resize(test_trgad, ((test_size, input_size, input_size)))
    test_regad = resize(test_regad, ((test_size, input_size, input_size)))

    test_gas = resize(test_gas, ((test_size, input_size, input_size)))
    test_trgas = resize(test_trgas, ((test_size, input_size, input_size)))
    test_regas = resize(test_regas, ((test_size, input_size, input_size)))

    test_mk = resize(test_mk, ((test_size, input_size, input_size)))
    test_trmk = resize(test_trmk, ((test_size, input_size, input_size)))
    test_remk = resize(test_remk, ((test_size, input_size, input_size)))

    test_ctwav = resize(test_ctwav, ((test_size, input_size, input_size)))

    test_dict = {"rp": test_rp, "rerp": test_rerp, "trrp": test_trrp, "gas": test_gas, "regas": test_regas,
                 "trgas": test_trgas, "gad": test_gad, "regad": test_regad, "trgad": test_trgad, "mk": test_mk,
                 "remk": test_remk, "trmk": test_trmk, "ctwav": test_ctwav}
    coding_name = ["rp", "rerp", "trrp", "gas", "regas", "trgas", "gad", "regad", "trgad", "mk", "remk", "trmk",
                   "ctwav"]

    for i in coding_name:
        test_dict[i] = np.expand_dims(test_dict[i], axis=1)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = "1"
    print(torch.cuda.is_available())
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("current cuda device:", torch.cuda.current_device())
    print("count of using gpus:", torch.cuda.device_count())

    stage_num = args.stage
    cell = args.cell
    dim = args.dim
    start_dim = copy.deepcopy(dim)
    cells = []
    dims = []

    for s in range(stage_num):
        if s == 0:
            cells = cell
            dims = dim
        else:
            add_cell = cell
            dim = 2*dim
            cells = np.append(cells, add_cell)
            dims = np.append(dims, dim)
    cells = cells.tolist()
    dims = dims.tolist()
    print("embed_dims : ", dims)
    print("cells : ", cells)
    transitions = [True, True, True, True]
    mlp_ratios = [4, 4, 4, 4]
    offset_size = [16, 8, 4, 2]
    infeature_list = args.imgcode
    print(infeature_list)
    channel_num = len(infeature_list)
    model = CycleNet(cells, channel_num, embed_dims=dims, patch_size=7, transitions=transitions,
                     num_classes=class_num,
                     mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, offset_size=offset_size).to(device)
    parameter_num = count_parameters(model)
    print("parameter_num : ", parameter_num)
    test_stack = stacking(input_size, test_size, channel_num, infeature_list)
    test_all = test_stack(test_dict)


    test_data = CustomDataset(test_all, ucr_target_test)
    testloader = DataLoader(test_data, batch_size=5, shuffle=False)
    # test_fix = np.concatenate(
    #     (test_dict["rp"], test_dict["rerp"], test_dict["trrp"], test_dict["gas"], test_dict["regas"]
    #      , test_dict["trgas"], test_dict["trgad"], test_dict["mk"], test_dict["remk"], test_dict["ctwav"]), axis=1)
    # best_test_acc = 0
    #
    # test_data = CustomDataset(test_fix, ucr_target_test)
    # testloader = DataLoader(test_data, batch_size=5, shuffle=False)
    model.load_state_dict(copy.deepcopy(torch.load(RPATH + 'Lightning264.pt')))
    # model.load_state_dict(copy.deepcopy(torch.load(RPATH +data_name+str(input_size)+'.pt')))
    # model.load_state_dict(torch.load(RPATH + data_name + str(input_size) + '.pt'))

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    accuracy = []
    iter_acc = []
    test_loss = []
    y_true_list = []
    y_pred_list = []
    b = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            b+=1
            data, labels = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            _, predicted =  torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter_acc.append(b)
            cost = 100 * correct / total
            accuracy.append(cost)
            y_true = labels.tolist()
            y_pred = predicted.tolist()
            y_true_list.extend(y_true)
            y_pred_list.extend(y_pred)
        test_loss.append(loss)
        error_rate = 1 - (correct / total)
        test_acc = 100. * correct / len(testloader.dataset)

    if class_num > 2:
        f1 = f1_score(y_true_list, y_pred_list, average='weighted')
        auc_score = 0
    elif class_num == 2:
        f1 = 0
        auc_score = roc_auc_score(y_true_list, y_pred_list)
    print("f1 :", f1)
    print("roc_auc_score :", auc_score)
    print('accuracy of testdata: %d %%' % (test_acc))
    print('error rate : ', format(error_rate, ".3f"))