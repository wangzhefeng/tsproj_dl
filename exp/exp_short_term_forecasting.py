# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_short_term_forecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-09
# * Version     : 1.0.060914
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.TFs_type.data_factory import data_provider
from utils.model_tools import EarlyStopping, adjust_learning_rate
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4 import M4Meta
from utils.m4_summary import M4Summary
from utils.model_memory import model_memory_size
from utils.plot_results import predict_result_visual
from utils.plot_losses import plot_losses
from utils.timestamp_utils import from_unix_time
from utils.log_util import logger


M4_FORECAST_GROUPS = {
    "Weekly_forecast.csv",
    "Monthly_forecast.csv",
    "Yearly_forecast.csv",
    "Daily_forecast.csv",
    "Hourly_forecast.csv",
    "Quarterly_forecast.csv",
}


class Exp_Short_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        # 时间序列模型初始化
        logger.info(f"Initializing model {self.args.model}...")
        model = self.get_model_class(self.args.model)(self.args).float()
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # 打印模型参数量
        model_memory_size(model, verbose=True)
        
        return model 
    
    def _get_data(self, flag: str):
        """
        数据集构建
        """
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader
    
    def _select_criterion(self):
        """
        评价指标
        """
        if self.args.loss == 'MSE':
            return nn.MSELoss()
        elif self.args.loss == 'MAPE':
            return mape_loss()
        elif self.args.loss == 'MASE':
            return mase_loss()
        elif self.args.loss == 'SMAPE':
            return smape_loss()
        elif self.args.loss == "L1":
            return nn.L1Loss()

    def _select_optimizer(self):
        """
        优化器
        """
        if self.args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError(f"unsupported optimizer: {self.args.optimizer}")

        return optimizer

    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = Path(self.args.checkpoints).joinpath(setting)
        model_path.mkdir(parents=True, exist_ok=True)
        # 最优模型保存路径
        model_checkpoint_path = model_path.joinpath("checkpoint.pth")
        
        return model_checkpoint_path

    def _get_test_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = Path(self.args.test_results).joinpath(setting)
        results_path.mkdir(parents=True, exist_ok=True)
        
        return results_path

    def _get_predict_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = Path(self.args.forecast_results).joinpath(setting)
        results_path.mkdir(parents=True, exist_ok=True)
        
        return results_path 

    # TODO
    def _test_results_save(self, preds, trues, setting, path):
        """
        测试结果保存
        """
        from utils.metrics_dl import metric
        # 计算测试结果评价指标
        r2, mse, rmse, mae, mape, mape_accuracy, mspe, dtw = metric(preds, trues, use_dtw=self.args.use_dtw)
        logger.info(f"Test results: r2:{r2:.4f} mse:{mse:.4f} rmse:{rmse:.4f} mae:{mae:.4f} mape:{mape:.4f} mape accuracy:{mape_accuracy:.4f} mspe:{mspe:.4f} dtw: {dtw}")
        
        # result1 保存
        with open(Path(path).joinpath("result_forecast.txt"), 'a', encoding="utf-8") as file:
            file.write(setting + "  \n")
            file.write(f"r2:{r2}, mse:{mse}, rmse:{rmse}, mae:{mae}, mape:{mape}, mape accuracy:{mape_accuracy}, mspe:{mspe}, dtw:{dtw}")
            file.write('\n')
            file.write('\n')
            file.close()
        # result2 保存
        np.save(
            Path(path).joinpath('metrics.npy'), 
            np.array([r2, mae, mse, rmse, mape, mape_accuracy, mspe], dtype=float)
        )
        np.save(Path(path).joinpath('preds.npy'), preds)
        np.save(Path(path).joinpath('trues.npy'), trues)
    
    def _pred_results_save(self, preds, preds_df, path):
        """
        预测结果保存
        """
        if preds is not None:
            np.save(Path(path).joinpath("prediction.npy"), preds) 
        if preds_df is not None:
            preds_df.to_csv(
                Path(path).joinpath("prediction.csv"), 
                encoding="utf_8_sig", 
                index=False
            )

    def _model_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark, flag):
        """
        短期预测前向传播。
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        def _run_model():
            outputs = self.model(batch_x, None, dec_inp, None)
            if self.args.output_attention and isinstance(outputs, (tuple, list)):
                return outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.amp.autocast("cuda"):
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:]

        if flag == "test":
            return outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy(), batch_y_mark.detach().cpu().numpy()
        return outputs, batch_y, batch_y_mark

    def _compute_loss(self, criterion, insample, forecast, target, mask):
        if self.args.loss in {"MAPE", "MASE", "SMAPE"}:
            return criterion(insample, self.args.frequency_map, forecast, target, mask)
        return criterion(forecast, target)

    def train(self, setting):
        # 数据集构建
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='valid')
        # checkpoint 保存路径
        logger.info(f"{40 * '-'}")
        logger.info(f"Model checkpoint will be saved in path:")
        logger.info(f"{40 * '-'}")
        model_checkpoint_path = self._get_model_path(setting)
        logger.info(model_checkpoint_path)
        # 测试结果保存地址
        logger.info(f"{40 * '-'}")
        logger.info(f"Train results will be saved in path:")
        logger.info(f"{40 * '-'}")
        test_results_path = self._get_test_results_path(setting) 
        logger.info(test_results_path)
        # 模型训练
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start training...")
        logger.info(f"{40 * '-'}")
        # time: 模型训练开始时间
        train_start_time = time.time()
        logger.info(f"Train start time: {from_unix_time(train_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        # 训练窗口数
        train_steps = len(train_loader)
        logger.info(f"Train total steps: {train_steps}") 
        # 模型优化器
        optimizer = self._select_optimizer()
        logger.info(f"Train optimizer has builded...")
        # 模型损失函数
        criterion = self._select_criterion()
        logger.info(f"Train criterion has builded...")
        # 早停类实例
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        logger.info(f"Train early stopping instance has builded, patience: {self.args.patience}")
        # learning rate scheduler
        if self.args.lradj == "TST":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate,
            )
        else:
            scheduler = None
        # 自动混合精度训练
        if self.args.use_amp:
            scaler = torch.amp.GradScaler()
        # 训练、验证结果收集
        train_losses, vali_losses = [], []
        # 分 epoch 训练
        for epoch in range(self.args.train_epochs):
            # time: epoch 训练开始时间
            epoch_start_time = time.time()
            logger.info(f"Epoch: {epoch+1} \tstart time: {from_unix_time(epoch_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            # epoch 训练结果收集
            iter_count = 0
            train_loss = []
            # 模型训练模式
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # logger.info(f"Train step: {i} running...")
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # 模型优化器梯度归零
                optimizer.zero_grad()
                # 前向传播
                outputs, batch_y, batch_y_mark = self._model_forward(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, flag="train"
                )
                # 计算损失
                loss = self._compute_loss(criterion, batch_x.float().to(self.device), outputs, batch_y, batch_y_mark)
                train_loss.append(loss.item())
                
                # 当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失
                if (i + 1) % 5 == 0:
                    speed = (time.time() - train_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'Epoch: {epoch + 1}, \tBatch: {i + 1} | train loss: {loss.item():.7f}, \tSpeed: {speed:.4f}s/batch; left time: {left_time:.4f}s')
                    iter_count = 0
                    train_start_time = time.time()
                # ------------------------------
                # 后向传播、参数优化更新
                # ------------------------------
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            
            # 日志打印: 训练 epoch、每个 epoch 训练的用时
            logger.info(f"Epoch: {epoch + 1}, \tCost time: {time.time() - epoch_start_time}")
            # 模型验证
            train_loss = np.average(train_loss)
            vali_loss = self.valid(train_loader, vali_loader, criterion)
            logger.info(f"Epoch: {epoch + 1}, \tSteps: {train_steps} | Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
            # 训练/验证损失收集
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            # 早停机制、模型保存
            early_stopping(
                epoch=epoch,
                val_loss=vali_loss,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_path=model_checkpoint_path,
            )
            if early_stopping.early_stop:
                logger.info(f"Epoch: {epoch + 1}, \tEarly stopping...")
                break
            # 学习率调整
            if self.args.lradj != "TST":
                adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args, printout=True)
            else:
                logger.info(f"Updating learning rate to {scheduler.get_last_lr()[0]}")
        # -----------------------------
        # 模型加载
        # ------------------------------
        logger.info(f"{40 * '-'}")
        logger.info(f"Training Finished!")
        logger.info(f"{40 * '-'}")
        # plot losses
        logger.info("Plot and save train/valid losses...")
        plot_losses(
            train_epochs=self.args.train_epochs,
            train_losses=train_losses, 
            vali_losses=vali_losses, 
            label="loss",
            results_path=test_results_path
        )
        # load model
        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device)["model"])
        # return model and train results
        logger.info("Return training results...")
        return self.model

    def valid(self, train_loader, vali_loader, criterion):
        """
        模型验证
        """
        # 模型开始验证
        logger.info(f"Model start validating...")
        # TODO 数据处理
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)
        # 模型验证结果
        # vali_loss = []
        # 模型评估模式
        self.model.eval()
        with torch.no_grad():
            logger.info(f"Valid step: running...")
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()#.to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            # 前向传播
            for i in range(len(id_list) - 1):
                model_outputs = self.model(
                    x[id_list[i]:id_list[i + 1]], 
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None
                )
                if self.args.output_attention and isinstance(model_outputs, (tuple, list)):
                    model_outputs = model_outputs[0]
                outputs[id_list[i]:id_list[i + 1], :, :] = model_outputs.detach().cpu()
            # 预测值/真实值提取
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y, dtype=np.float32))
            batch_y_mark = torch.ones(true.shape)
            # 计算/保存验证损失
            loss = self._compute_loss(criterion, x.detach().cpu()[:, :, 0], pred[:, :, 0], true, batch_y_mark)
            # vali_loss.append(loss)
            # logger.info(f"debug::valid step: {i}, valid loss: {loss.item()}")
        # 计算模型输出
        self.model.train()
        # log
        logger.info(f"Validating Finished!")
        return float(loss.item())

    def test(self, setting, load: bool=False):
        # 数据集构建
        _, train_loader = self._get_data(flag = 'train')
        _, test_loader = self._get_data(flag = 'test')
        # TODO 数据处理
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)
        # 模型加载
        if load:
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device)["model"]) 
            logger.info(model_checkpoint_path)
        # 测试结果保存地址
        logger.info(f"{40 * '-'}")
        logger.info(f"Test results will be saved in path:")
        logger.info(f"{40 * '-'}")
        test_results_path = self._get_test_results_path(setting) 
        logger.info(test_results_path) 
        # 模型开始测试
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start testing...")
        logger.info(f"{40 * '-'}")
        # ------------------------------
        # 模型推理
        # ------------------------------
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        preds, trues = [], []
        preds_flat, trues_flat = [], []
        with torch.no_grad():
            logger.info(f"Test step: running...")
            # dec input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 500)
            id_list = np.append(id_list, B)
            # 前向传播
            for i in range(len(id_list) - 1):
                model_outputs = self.model(
                    x[id_list[i]:id_list[i + 1]], 
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]], 
                    None
                )
                if self.args.output_attention and isinstance(model_outputs, (tuple, list)):
                    model_outputs = model_outputs[0]
                outputs[id_list[i]:id_list[i + 1], :, :] = model_outputs
                if id_list[i] % 1000 == 0:
                    logger.info(f"id_list[i]: {id_list[i]}")
            # 预测值/真实值提取
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()
            # 测试结果收集
            preds = outputs
            trues = np.array(y, dtype=np.float32)
            # 预测数据可视化
            x = x.detach().cpu().numpy()
            plot_step = max(1, preds.shape[0] // 10)
            for i in range(0, preds.shape[0], plot_step):
                true_plot = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pred_plot = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                predict_result_visual(pred_plot, true_plot, path=Path(test_results_path).joinpath(f"{i}.pdf"))
        # 测试结果保存
        folder_path = Path("./m4_results").joinpath(self.args.model)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{40 * '-'}")
        logger.info(f"Test metric results have been saved in path:")
        logger.info(f"{40 * '-'}")
        forecasts_df = pd.DataFrame(
            preds[:, :, 0], 
            columns=[f'V{i + 1}' for i in range(self.args.pred_len)]
        )
        forecasts_df.insert(0, "id", test_loader.dataset.ids[:preds.shape[0]])
        forecast_path = folder_path.joinpath(f"{self.args.seasonal_patterns}_forecast.csv")
        forecasts_df.to_csv(forecast_path, index=False, encoding="utf-8")
        self._test_results_save(preds[:, :, 0], trues, setting, test_results_path)
        logger.info(forecast_path)
        logger.info(test_results_path)
        
        if M4_FORECAST_GROUPS.issubset({path.name for path in folder_path.iterdir()}):
            m4_summary = M4Summary(f"{folder_path.as_posix()}/", self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            logger.info(f"Test results: smape:{smape_results} mape:{mape} mase:{mase} owa:{owa_results}")
        else:
            logger.info('After all 6 tasks are finished, you can calculate the averaged index')
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
