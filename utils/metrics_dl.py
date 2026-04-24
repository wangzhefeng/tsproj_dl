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
from typing import Union, List

import numpy as np
from sklearn.metrics import r2_score

from utils.dtw_metric import accelerated_dtw
from utils.log_util import logger


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    
    return (u / d).mean(-1)


def R_square(pred, true):
    return float(r2_score(true, pred))


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
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = np.where(np.abs(true) < 1e-8, 1.0, np.abs(true))
    return float(np.mean(np.abs((true - pred) / denominator)))


def Accuracy(pred, true):
    """
    时序预测准确率计算，1-MAPE
    """
    return 1 - MAPE_v2(true, pred)


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def DTW(preds, trues, use_dtw=False):
    if use_dtw:
        dtw_list = []
        manhattan_distance = lambda x, y: np.abs(x - y)
        for i in range(preds.shape[0]):
            x = preds[i].reshape(-1, 1)
            y = trues[i].reshape(-1, 1)
            if i % 100 == 0:
                logger.info(f"calculating dtw iter: {i}")
            d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
            dtw_list.append(d)
        dtw = np.array(dtw_list).mean()
    else:
        dtw = "Not calculated"
    
    return dtw


def cal_accuracy(y_pred, y_true):
    """
    计算准确率
    """
    return np.mean(y_pred == y_true)


def metric(pred, true, use_dtw=False):
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)
    r2 = R_square(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mae = MAE(pred, true)
    mape = MAPE(pred, true)
    accuracy = Accuracy(pred, true)
    mspe = MSPE(pred, true)
    if use_dtw:
        dtw = DTW(pred, true, use_dtw)
        return (r2, mse, rmse, mae, mape, accuracy, mspe, dtw)
    else:
        return (r2, mse, rmse, mae, mape, accuracy, mspe, "Not calculated")






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
    
    target = np.array([1, 10, 1e6], dtype=float)
    preds = np.array([0.9, 15, 1.2e6], dtype=float)
    r2 = r2_score(target, preds)
    print(r2)
    mape = MAPE_v2(target, preds)
    print(mape)
    print(1-mape)

if __name__ == "__main__":
    main()
