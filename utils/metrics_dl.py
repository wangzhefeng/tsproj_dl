# -*- coding: utf-8 -*-

# ***************************************************
# * File        : metrics_dl.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-02
# * Version     : 0.1.110215
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import Union, List

import numpy as np
import torch
from torchmetrics.functional.regression import (
    r2_score,
    mean_absolute_percentage_error, 
)

from utils.dtw_metric import accelerated_dtw

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    
    return (u / d).mean(-1)

def R_square(pred, true):
    pred = torch.from_numpy(pred)
    true = torch.from_numpy(true)
    # true, pred = np.array(true), np.array(pred)
    return r2_score(pred, true)


def MSE(pred, true):
    """
    Calculates MSE(mean squared error) given true and pred
    """
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MAPE_v2(true: Union[List, np.array], pred: Union[List, np.array]):
    """
    Calculates MAPE(mean absolute percentage error) given true and pred
    """
    pred = torch.from_numpy(pred)
    true = torch.from_numpy(true)
    return mean_absolute_percentage_error(pred, true)


def Accuracy(pred, true):
    """
    时序预测准确率计算，1-MAPE
    """
    pred = torch.from_numpy(pred)
    true = torch.from_numpy(true)
    
    return 1 - mean_absolute_percentage_error(pred, true)


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def DTW(preds, trues):
    dtw_list = []
    manhattan_distance = lambda x, y: np.abs(x - y)
    for i in range(preds.shape[0]):
        x = preds[i].reshape(-1,1)
        y = trues[i].reshape(-1,1)
        if i % 100 == 0:
            print("calculating dtw iter:", i)
        d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
        dtw_list.append(d)
    dtw = np.array(dtw_list).mean()
    
    return dtw


def metric(pred, true):
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)
    r2 = R_square(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mae = MAE(pred, true)
    mape = MAPE(pred, true)
    accuracy = Accuracy(pred, true)
    mspe = MSPE(pred, true)
    # dtw = DTW(pred, true)

    return r2, mse, rmse, mae, mape, accuracy, mspe


def cal_accuracy(y_pred, y_true):
    """
    计算准确率
    """
    return np.mean(y_pred == y_true)




# 测试代码 main 函数
def main():
    # np.random.seed(0)
    # y_true = np.random.rand(10)
    # y_pred = np.random.rand(10)
    # print(y_true)
    # print(y_pred)
    # mae, mse, rmse, mape, accuracy, mspe = metric(y_pred, y_true)
    # print(f"mae: {mae}\nmse: {mse}\nrmse: {rmse}\nmape: {mape}\
    #     \naccuracy: {accuracy}\nmspe: {mspe}")
    
    import torch
    from torchmetrics.functional.regression import mean_absolute_percentage_error
    
    target = torch.tensor([1, 10, 1e6])
    preds = torch.tensor([0.9, 15, 1.2e6])
    r2 = r2_score(preds, target)
    print(r2)
    mape = mean_absolute_percentage_error(preds, target)
    print(mape)
    print(1-mape)

if __name__ == "__main__":
    main()
