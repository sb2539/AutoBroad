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

if __name__ == "__main__":
    # args = parser.parse_args()
    PATH = "/home/sin/PycharmProjects/please/save/FaceFour/"
    #RPATH = PATH + args.ucr + "/"
    print(PATH)
    #print(RPATH)
    data_name = "FaceFour"
    ucr = fetch_ucr_dataset(data_name, use_cache=True, data_home=None, return_X_y=False)
    ucr_train = ucr.data_train
    ucr_target_train = ucr.target_train
    ucr_test = ucr.data_test
    ucr_target_test = ucr.target_test
    ucr_train_re = resize(ucr_train, ((24, 350, 1)))
    ucr_test_re = resize(ucr_test, ((88, 350, 1)))
    trencoder = LabelEncoder()
    trencoder.fit(ucr_target_train)
    ucr_target_train = trencoder.transform(ucr_target_train)
    teencoder = LabelEncoder()
    teencoder.fit(ucr_target_test)
    ucr_target_test = teencoder.transform((ucr_target_test))

    class_num = 4
    input_size = 64
    train_size = len(ucr_train_re)
    test_size = len(ucr_test_re)

    tran = Image_coding(decompose=True)
    train_rp, train_rerp, train_trrp, train_gas, train_regas, train_trgas, train_gad, train_regad, train_trgad, train_mk, train_remk, train_trmk, train_ctwav = tran(
        ucr_train_re)
    test_rp, test_rerp, test_trrp, test_gas, test_regas, test_trgas, test_gad, test_regad, test_trgad, test_mk, test_remk, test_trmk, test_ctwav = tran(
        ucr_test_re)

    train_rp = resize(train_rp, ((train_size, input_size, input_size)))
    test_rp = resize(test_rp, ((test_size, input_size, input_size)))
    train_trrp = resize(train_trrp, ((train_size, input_size, input_size)))
    test_trrp = resize(test_trrp, ((test_size, input_size, input_size)))
    train_rerp = resize(train_rerp, ((train_size, input_size, input_size)))
    test_rerp = resize(test_rerp, ((test_size, input_size, input_size)))

    train_gad = resize(train_gad, ((train_size, input_size, input_size)))
    test_gad = resize(test_gad, ((test_size, input_size, input_size)))
    train_trgad = resize(train_trgad, ((train_size, input_size, input_size)))
    test_trgad = resize(test_trgad, ((test_size, input_size, input_size)))
    train_regad = resize(train_regad, ((train_size, input_size, input_size)))
    test_regad = resize(test_regad, ((test_size, input_size, input_size)))

    train_gas = resize(train_gas, ((train_size, input_size, input_size)))
    test_gas = resize(test_gas, ((test_size, input_size, input_size)))
    train_trgas = resize(train_trgas, ((train_size, input_size, input_size)))
    test_trgas = resize(test_trgas, ((test_size, input_size, input_size)))
    train_regas = resize(train_regas, ((train_size, input_size, input_size)))
    test_regas = resize(test_regas, ((test_size, input_size, input_size)))

    train_mk = resize(train_mk, ((train_size, input_size, input_size)))
    test_mk = resize(test_mk, ((test_size, input_size, input_size)))
    train_trmk = resize(train_trmk, ((train_size, input_size, input_size)))
    test_trmk = resize(test_trmk, ((test_size, input_size, input_size)))
    train_remk = resize(train_remk, ((train_size, input_size, input_size)))
    test_remk = resize(test_remk, ((test_size, input_size, input_size)))

    train_ctwav = resize(train_ctwav, ((train_size, input_size, input_size)))
    test_ctwav = resize(test_ctwav, ((test_size, input_size, input_size)))

    train_dict = {"rp": train_rp, "rerp": train_rerp, "trrp": train_trrp, "gas": train_gas, "regas": train_regas,
                  "trgas": train_trgas, "gad": train_gad, "regad": train_regad, "trgad": train_trgad, "mk": train_mk,
                  "remk": train_remk, "trmk": train_trmk, "ctwav": train_ctwav}
    test_dict = {"rp": test_rp, "rerp": test_rerp, "trrp": test_trrp, "gas": test_gas, "regas": test_regas,
                 "trgas": test_trgas, "gad": test_gad, "regad": test_regad, "trgad": test_trgad, "mk": test_mk,
                 "remk": test_remk, "trmk": test_trmk, "ctwav": test_ctwav}
    coding_name = ["rp", "rerp", "trrp", "gas", "regas", "trgas", "gad", "regad", "trgad", "mk", "remk", "trmk",
                   "ctwav"]

    for i in coding_name:
        train_dict[i] = np.expand_dims(train_dict[i], axis=1)
        test_dict[i] = np.expand_dims(test_dict[i], axis=1)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = "1"
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("current cuda device:", torch.cuda.current_device())
    print("count of using gpus:", torch.cuda.device_count())

    train_fix = np.concatenate((train_dict["rp"],train_dict["rerp"],train_dict["trrp"],train_dict["gas"]
                                ,train_dict["regas"],train_dict["trgas"],train_dict["gad"],train_dict["regad"],train_dict["trgad"],train_dict["mk"],train_dict["remk"],train_dict["trmk"],train_dict["ctwav"]), axis=1)
    test_fix = np.concatenate((test_dict["rp"],test_dict["rerp"],test_dict["trrp"],test_dict["gas"]
                               ,test_dict["regas"],test_dict["trgas"],test_dict["gad"],test_dict["regad"],test_dict["trgad"],test_dict["mk"],test_dict["remk"],test_dict["trmk"],test_dict["ctwav"]), axis=1)
    # train_fix = train_dict["gad"]
    # test_fix = test_dict["gad"]
    best_test_acc = 0

    stage_num = 4
    cell = 1
    dim = 16
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
    print("infeature_list: rp trgas mk trmk")
    channel_num = 13


    train_data = CustomDataset(train_fix, ucr_target_train)
    trainloader = DataLoader(train_data, batch_size=5, shuffle=True)
    test_data = CustomDataset(test_fix, ucr_target_test)
    testloader = DataLoader(test_data, batch_size=5, shuffle=False)
    #model.load_state_dict(copy.deepcopy(torch.load(RPATH +data_name+str(input_size)+'.pt')))

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    correct = 0
    total = 0
    img_test_acc = 0
    num_save = 0

    # y_true_list = []
    # y_pred_list = []
    b = 0
    loss_list = []
    iter = []
    train_accuracy = []
    epochs = 90
    for i in range(2000):
        y_true_list = []
        y_pred_list = []
        model = CycleNet(cells, channel_num, embed_dims=dims, patch_size=7, transitions=transitions,
                         num_classes=class_num, mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, offset_size=offset_size).to(device)
        parameter_num = count_parameters(model)
        print("parameter_num : ", parameter_num)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model.train()
        print("epoch : ", epochs)
        print("==============================================================")
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            for data in trainloader:
                input, target = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(output.data, 1)
                correct += preds.eq(target).sum().item()
                running_loss += loss.item()
        training_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / len(trainloader.dataset)
        print(
            "=============================================================================================================================================================")
        print(
            f"TRAIN: EPOCH {epoch + 1:04d} / {epochs:04d} | Epoch LOSS {training_loss:.4f} | Epoch ACC {train_acc:.4f} ")
        cost = running_loss / 6
            #torch.save(model.state_dict(), PATH + str(i) + 'ScreenType＿train.pt')
        loss_list.append(training_loss)
        train_accuracy.append(train_acc)
        running_loss = 0.0
        print('finish')
        correct = 0
        total = 0
        accuracy = []
        iter_acc = []
        test_loss = []
        model.eval()
        with torch.no_grad():
            for data in testloader:
                b+=1
                data, labels = data[0].to(device), data[1].to(device)
                outputs = model(data)
                #loss = criterion(outputs, labels)
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
            #test_loss.append(loss)
            error_rate = 1 - (correct / total)
            test_acc = 100. * correct / len(testloader.dataset)

        if class_num > 2:
            f1 = f1_score(y_true_list, y_pred_list, average='weighted')
            auc_score = 0
        elif class_num == 2:
            f1 = 0
            auc_score = roc_auc_score(y_true_list, y_pred_list)
        if img_test_acc < test_acc:
            num_save = num_save+1
            img_test_acc = test_acc
            torch.save(model.state_dict(), PATH +str(num_save) +data_name+'128(new).pt')
            print("save best test")
        print("f1 :", f1)
        print("roc_auc_score :", auc_score)
        print('accuracy of testdata: %d %%' % (test_acc))
        print('error rate : ', format(error_rate, ".3f"))
        print("best_test_acc : ", img_test_acc)


    model.load_state_dict(torch.load(PATH +str(num_save) +data_name + '128(new).pt'))
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    accuracy = []
    iter_acc = []
    test_loss = []
    y_true_list = []
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for data in testloader:
            b += 1
            data, labels = data[0].to(device), data[1].to(device)
            outputs = model(data)
            #loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter_acc.append(b)
            cost = 100 * correct / total
            accuracy.append(cost)
            y_true = labels.tolist()
            y_pred = predicted.tolist()
            y_true_list.extend(y_true)
            y_pred_list.extend(y_pred)
        #test_loss.append(loss)
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
    print("best_test_acc : ", img_test_acc)