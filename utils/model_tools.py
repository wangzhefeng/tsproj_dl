# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-13
# * Version     : 1.0.011322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math

import numpy as np
import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    """
    学习率调整
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2:  5e-5, 4:  1e-5, 6:  5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8,
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}  
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == 'type6':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == 'type7':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == "PEMS":
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        lr_adjust = {epoch: args.learning_rate * (0.2 ** (epoch // 2))}
    # update learning rate
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            logger.info(f'Epoch: {epoch}, \tUpdating learning rate to {lr}.')


class EarlyStopping:
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, epoch, val_loss, model, optimizer=None, scheduler=None, model_path=""):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer, scheduler, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"Epoch: {epoch+1}, \tEarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer, scheduler, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, optimizer, scheduler, model_path):
        # 日志打印
        if self.verbose:
            logger.info(f"\tEpoch: {epoch+1}, \tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        # 模型保存
        training_state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optmizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(training_state, model_path)
        self.val_loss_min = val_loss


def adjustment(gt, pred):
    """
    异常检测
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
            
    return gt, pred




# 测试代码 main 函数
def main():
   class Config:
       train_epochs = 10
       learning_rate = 1e-3
       lradj = 'type1'

   config = Config()
   
   adjust_learning_rate(optimizer=None, epoch=1, args=config)
       
if __name__ == "__main__":
   main()
