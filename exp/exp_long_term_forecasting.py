import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path: sys.path.append(ROOT)
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from data_provider.TFs_type.data_factory import data_provider
from utils.metrics_dl import metric
from utils.model_tools import adjust_learning_rate, EarlyStopping
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.plot_results import predict_result_visual
from utils.plot_losses import plot_losses
from utils.model_memory import model_memory_size
from utils.timestamp_utils import from_unix_time
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Exp_Long_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        # 时间序列模型初始化
        logger.info(f"Initializing model {self.args.model}...")
        model = self.get_model_module(self.args.model).Model(self.args).float()
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
        if self.args.loss == "MSE":
            return nn.MSELoss()
        elif self.args.loss == "MAPE":
            return mape_loss()
        elif self.args.loss == "MASE":
            return mase_loss()
        elif self.args.loss == "SMAPE":
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
        
        return optimizer
    
    def _get_model_path(self, setting):
        """
        模型保存路径，如果进行模型训练任务，则需要保存模型
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

    def _test_results_save(self, preds, trues, setting, path):
        """
        测试结果保存
        """
        # 计算测试结果评价指标
        r2, mse, rmse, mae, mape, mape_accuracy, mspe = metric(preds, trues)
        dtw = DTW(preds, trues) if self.args.use_dtw else -999
        logger.info(f"Test results: r2:{r2:.4f} mse:{mse:.4f} rmse:{rmse:.4f} mae:{mae:.4f} mape:{mape:.4f} mape accuracy:{mape_accuracy:.4f} mspe:{mspe:.4f} dtw: {dtw:.4f}")
        # result1 保存
        with open(Path(path).joinpath("result_forecast.txt"), 'a') as file:
            file.write(setting + "  \n")
            file.write(f"r2:{r2:.4f}, mse:{mse:.4f}, rmse:{rmse:.4f}, mae:{mae:.4f}, mape:{mape:.4f}, mape accuracy:{mape_accuracy:.4f}, mspe:{mspe:.4f}, dtw:{dtw:.4f}")
            file.write('\n')
            file.write('\n')
            file.close()
        # result2 保存
        np.save(
            Path(path).joinpath('metrics.npy'), 
            np.array([r2, mae, mse, rmse, mape, mape_accuracy, mspe, dtw])
        )
        np.save(Path(path).joinpath('preds.npy'), preds)
        np.save(Path(path).joinpath('trues.npy'), trues)
    
    def _pred_results_save(self, trues_df, preds_df, preds=None, path="./", setting=None):
        """
        预测结果保存
        """
        if preds is not None:
            np.save(Path(path).joinpath("prediction.npy"), preds) 

        if trues_df is not None:
            trues_df.to_csv(path.joinpath('history.csv'), index=False, encoding="utf_8_sig")
        
        if preds_df is not None:
            preds_df.to_csv(path.joinpath('forecast.csv'), index=False, encoding="utf_8_sig")
        
        with open(path.joinpath('summary.txt'), 'w', encoding='utf-8') as summary_file:
            summary_file.write(setting + '\n')
            summary_file.write(f'prediction only: no ground truth available\n')
            summary_file.write(f'history_points:{len(trues_df)}, forecast_points:{len(preds_df)}\n')
            summary_file.write(f'forecast_target:{preds_df.columns[-1]}\n')
    
    def _model_forward(self, data, batch_x, batch_y, batch_x_mark, batch_y_mark, flag, reverse=False):
        """
        前向传播
        """
        # 数据预处理
        # ---------------------
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        # logger.info(f"debug::batch_x: {batch_x.shape}")
        # logger.info(f"debug::batch_x_mark: {batch_x_mark.shape}")
        # logger.info(f"debug::batch_y: {batch_y.shape}")
        # logger.info(f"debug::batch_y_mark: {batch_y_mark.shape}")
        
        # decoder input
        # ---------------------
        if batch_y.shape[1] != (self.args.label_len + self.args.pred_len):
            if flag in ["train", "valid", "test"]:
                logger.info(f"Train Stop::Data batch_y.shape[1] not equal to (label_len + pred_len).")
                return None, None
            elif flag == "pred":
                dec_inp = torch.zeros((batch_y.shape[0], self.args.pred_len, batch_y.shape[2]), dtype=batch_x.dtype, device=self.device)
        else:
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        if self.args.down_sampling_layers != 0:
            dec_inp = None
        # encoder-decoder
        # ---------------------
        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[0] if self.args.output_attention and isinstance(outputs, (tuple, list)) else outputs
            return outputs
        if self.args.use_amp:
            with torch.amp.autocast("cuda"):
                outputs = _run_model()
        else:
            outputs = _run_model()
        
        # pred and true process
        # ---------------------
        # pred and true 提取
        outputs = outputs[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :]
        # output detach device
        if flag in ["valid", "test", "pred"]:
            outputs = outputs.detach().cpu()
            batch_y = batch_y.detach().cpu()
        # 输入输出逆转换
        if reverse:
            outputs, batch_y = self._inverse_data(data, outputs, batch_y)
        # 预测值/真实值提取
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]
        if flag == "train":
            batch_y.to(self.device)
        
        return outputs, batch_y

    def _inverse_data(self, data, outputs, batch_y):
        """
        输入输出逆转换
        """
        if data.scale and self.args.inverse:
            outputs = outputs.numpy()
            batch_y = batch_y.numpy()
            # 数据逆转换 output 最后一个维度转换为与 batch_y 一致: [1, pred_len, enc_in/dec_in]
            if outputs.shape[-1] != batch_y.shape[-1]:
                outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
            # inverse transform
            shape = outputs.shape  # [batch, pred_len, enc_in/dec_in]
            outputs = data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
            batch_y = data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
            # or
            # outputs = data.inverse_transform(outputs.squeeze(0)).reshape(shape)
            # batch_y = data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

        return outputs, batch_y
    
    def train(self, setting):
        """
        模型训练
        """
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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )
        # 自动混合精度训练
        if self.args.use_amp:
            scaler = torch.amp.GradScaler()
        # 训练、验证结果收集
        train_losses, vali_losses = [], []
        # 分 epoch 训练
        for epoch in range(self.args.train_epochs):
            # time: epoch 训练开始时间
            epoch_start_time = time.time()
            logger.info(" ")
            logger.info(f"Epoch: {epoch+1}, \tstart time: {from_unix_time(epoch_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            # epoch 训练结果收集
            iter_count = 0
            train_loss = []
            # 模型训练模式
            self.model.train()
            for iters, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # 模型优化器梯度归零
                optimizer.zero_grad()
                # 前向传播
                outputs, batch_y = self._model_forward(
                    train_data,
                    batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    flag="train", reverse=False,
                )
                if outputs is None and batch_y is None: break
                # 计算训练损失
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                # 当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失
                if (iters + 1) % 50 == 0:
                    # 训练速度和时间
                    speed = (time.time() - train_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - iters)
                    logger.info(f'Epoch: {epoch + 1}, \tIters: {iters + 1} | train loss: {loss.item():.7f}, \tSpeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    # 打印模型参数量
                    # model_memory_size(self.model, verbose=True)
                    # 内存占用
                    # allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                    # cached_memory = torch.cuda.memory_reserved() / (1024 ** 2)
                    # logger.info(f"\t\tAllocated tensor emory: {allocated_memory:.1f}MB. Cached total memory: {cached_memory:.1f}MB.")
                    # 更新计数器和时间
                    iter_count = 0
                    train_start_time = time.time()
                # 后向传播、参数优化更新
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                # learning rate update
                if self.args.lradj != "TST":
                    adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            logger.info(f"Epoch: {epoch + 1}, \tCost time: {time.time() - epoch_start_time}")
            # 模型验证
            train_loss = np.average(train_loss)
            vali_loss = self.valid(vali_data, vali_loader, criterion)
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
        logger.info(f"{40 * '-'}")
        logger.info(f"Training Finished!")
        logger.info(f"{40 * '-'}")
        # plot train and valid losses
        logger.info("Plot and save train/valid losses...")
        plot_losses(self.args.train_epochs, train_losses, vali_losses, "loss", test_results_path)
        # load model
        logger.info("Loading best model...")
        self.model.load_state_dict(torch.load(model_checkpoint_path)["model"])
        # return model and train results
        logger.info("Return training results...")
        return self.model

    def valid(self, vali_data, vali_loader, criterion):
        """
        模型验证
        """
        # 模型开始验证
        logger.info(f"Model start validating...")
        # 验证窗口数
        vali_steps = len(vali_loader)
        logger.info(f"Vali total steps: {vali_steps}")
        # 模型验证结果
        vali_loss = []
        # 模型评估模式
        self.model.eval()
        with torch.no_grad():
            for iters, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                logger.info(f"Vali step: {iters} running...")
                # 前向传播
                outputs, batch_y = self._model_forward(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    flag="valid", reverse=False
                )
                if outputs is None and batch_y is None: break
                # 计算/保存验证损失
                loss = criterion(outputs, batch_y)
                vali_loss.append(loss)
        # 计算验证集上所有 batch 的平均验证损失
        vali_loss = np.average(vali_loss)
        # 计算模型输出
        self.model.train()
        # log
        logger.info(f"Validating Finished!")
        return vali_loss

    def test(self, setting, load: bool=False):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag="test")
        # 模型加载
        if load:
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path)["model"]) 
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
        # 模型测试次数
        test_steps = len(test_loader)
        logger.info(f"Test total steps: {test_steps}")
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        preds, trues = [], []
        preds_flat, trues_flat = [], []
        with torch.no_grad():
            for iters, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                logger.info(f"Test step: {iters} running...")
                # 前向传播
                outputs, batch_y = self._model_forward(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    flag = "test", reverse=True
                )
                if outputs is None and batch_y is None: break
                # 测试结果收集
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                # 预测数据可视化
                if iters % 10 == 0:
                    inputs = batch_x.numpy()
                    if test_data.scale and self.args.inverse:
                        inputs = test_data.inverse_transform(inputs.reshape(inputs.shape[0] * inputs.shape[1], -1)).reshape(inputs.shape)
                        # or inputs = test_data.inverse_transform(inputs.squeeze(0)).reshape(inputs.shape)
                    true_plot = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    pred_plot = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    predict_result_visual(pred_plot, true_plot, test_results_path, iters)
        # 测试结果保存
        logger.info(f"{40 * '-'}")
        logger.info(f"Test metric results have been saved in path:")
        logger.info(f"{40 * '-'}")
        preds = np.concatenate(preds, axis = 0)  # preds = np.array(preds)
        trues = np.concatenate(trues, axis = 0)  # trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # 结果收集
        logger.info(f"preds.reshape(1, -1): \n{preds.reshape(1, -1)}")
        logger.info(f"trues.reshape(1, -1): \n{trues.reshape(1, -1)}")
        logger.info(f"trues.reshape(1, -1)[0]: \n{trues.reshape(1, -1)[0]}")
        logger.info(f"trues.reshape(1, -1)[0]: \n{len(trues.reshape(1, -1)[0])}")
        test_results = pd.DataFrame({
            "preds": preds.reshape(1, -1)[0],
            "trues": trues.reshape(1, -1)[0],
        }, index=range(len(preds.reshape(1, -1)[0])))
        test_results.to_csv(Path(test_results_path).joinpath("test_results.csv"), index=False, encoding="utf-8")
        logger.info(f"test_results: \n{test_results}")
        self._test_results_save(preds.reshape(-1, 1), trues.reshape(-1, 1), setting, test_results_path)
        logger.info(test_results_path)
        # 测试结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Test visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        if self.args.features == 'M':
            preds_flat = np.concatenate(preds, axis = 0)[:, -1]
            trues_flat = np.concatenate(trues, axis = 0)[:, -1]
        else:
            preds_flat = np.concatenate(preds, axis = 0)
            trues_flat = np.concatenate(trues, axis = 0)
        predict_result_visual(preds_flat, trues_flat, path=test_results_path, iters=None) 
        logger.info(test_results_path)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return
    
    @staticmethod
    def _stitch_window_predictions(preds: np.ndarray, trues: np.ndarray):
        """
        将滑动窗口预测结果还原成时间轴上的连续序列。

        当前统一数据层的 test loader 默认使用 stride=1，
        直接 reshape/concatenate 会把重叠窗口重复拼接，导致时间顺序失真。
        这里按时间位置对所有重叠预测取均值，恢复真实时间轴。
        """
        num_windows, pred_len, channels = preds.shape
        stitched_len = num_windows + pred_len - 1

        pred_sum = np.zeros((stitched_len, channels), dtype=np.float64)
        true_sum = np.zeros((stitched_len, channels), dtype=np.float64)
        counts = np.zeros((stitched_len, 1), dtype=np.int64)

        for window_idx in range(num_windows):
            start = window_idx
            end = window_idx + pred_len
            pred_sum[start:end] += preds[window_idx]
            true_sum[start:end] += trues[window_idx]
            counts[start:end] += 1

        counts_safe = np.where(counts == 0, 1, counts)
        stitched_preds = pred_sum / counts_safe
        stitched_trues = true_sum / counts_safe

        return stitched_preds.astype(np.float32), stitched_trues.astype(np.float32), counts.squeeze(-1)

    @staticmethod
    def _build_stitched_results_frame(stitched_preds: np.ndarray, stitched_trues: np.ndarray, overlap_counts=None, stitched_dates=None):
        rows = {"step": np.arange(len(stitched_preds))}
        if stitched_dates is not None:
            rows["date"] = stitched_dates.astype(str)
        if overlap_counts is not None:
            rows["overlap_count"] = overlap_counts

        if stitched_preds.shape[1] == 1:
            rows["preds"] = stitched_preds[:, 0]
            rows["trues"] = stitched_trues[:, 0]
        else:
            for channel_idx in range(stitched_preds.shape[1]):
                rows[f"preds_{channel_idx}"] = stitched_preds[:, channel_idx]
                rows[f"trues_{channel_idx}"] = stitched_trues[:, channel_idx]

        return pd.DataFrame(rows)

    def _build_test_stitched_dates(self, test_data, stitched_len: int):
        """
        构建测试集重建时间轴。
        """
        segment_dates = getattr(test_data, "segment_dates", None)
        if segment_dates is None:
            return None
        stitched_dates = pd.Series(segment_dates).iloc[test_data.seq_len:test_data.seq_len + stitched_len].reset_index(drop=True)

        if len(stitched_dates) != stitched_len:
            return None

        return stitched_dates.to_numpy()

    def forecast(self, setting, load: bool=True):
        """
        模型预测
        https://snu77.blog.csdn.net/article/details/132881996
        https://github.com/thuml/Autoformer/blob/main/exp/exp_main.py#L241
        https://github.com/thuml/Autoformer/blob/main/predict.ipynb
        """
        # 构建预测数据集
        pred_data, pred_loader = self._get_data(flag='pred')
        # TODO
        # batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(pred_loader))
        # batch_x = batch_x.float().to(self.device)
        # batch_y = batch_y.float().to(self.device)
        # batch_x_mark = batch_x_mark.float().to(self.device)
        # batch_y_mark = batch_y_mark.float().to(self.device)
        # dec_zeros = torch.zeros((batch_x.shape[0], self.args.pred_len, batch_x.shape[-1]), dtype=batch_x.dtype, device=self.device)
        # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1).float().to(self.device)
        
        # 模型加载
        if load:
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
            model_checkpoint_path = self._get_model_path(setting)
            self.model.load_state_dict(torch.load(model_checkpoint_path)["model"]) 
            logger.info(model_checkpoint_path)
        # 模型预测结果保存地址
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast results will be saved in path:")
        logger.info(f"{40 * '-'}")
        pred_results_path = self._get_predict_results_path(setting)
        logger.info(pred_results_path)
        # 模型开始预测
        logger.info(f"{40 * '-'}")
        logger.info(f"Model start forecasting...")
        logger.info(f"{40 * '-'}")
        # 模型预测(推理)次数
        pred_steps = len(pred_loader)
        logger.info(f"Forecast total steps: {pred_steps}")
        # 模型评估模式
        self.model.eval()
        # 模型预测
        preds = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                logger.info(f"Forecast step: {i} running...")
                # 前向传播
                outputs, batch_y = self._model_forward(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, flag = "pred")
                
                # 输入输出逆转换
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs.numpy()  # [1, pred_len, 1]
                batch_y = batch_y.numpy()  # [1, pred_len, enc_in/dec_in]
                inputs = batch_x.detach().cpu().numpy()[:, :, f_dim:]  # [1, seq_len, 1]
                if pred_data.scale and self.args.inverse:
                    if self.args.features == 'MS':
                        outputs = pred_data.inverse_transform_target(outputs)
                        inputs = pred_data.inverse_transform_target(inputs)
                    else:
                        outputs = pred_data.inverse_transform_full(outputs)
                        inputs = pred_data.inverse_transform_full(inputs)
                # 预测结果收集
                preds.append(outputs)
                trues_plot = inputs[0, :, -1]
                preds_plot = np.concatenate((inputs[0, :, -1], outputs[0, :, -1]), axis=0)
                # logger.info(f"debug::trues_plot: \n{trues_plot} \ntrues_plot.shape: {trues_plot.shape}")
                # logger.info(f"debug::preds_plot: \n{preds_plot} \npreds_plot.shape: {preds_plot.shape}")
        # 最终预测值
        preds = np.array(preds).squeeze()
        logger.info(f"Forecast results: preds: \n{preds} \npreds shape: {preds.shape}")
        preds_df = pd.DataFrame({
            "timestamp": pd.date_range(pred_data.forecast_start_time, periods=self.args.pred_len, freq=self.args.freq),
            "predict_value": preds,
        })
        logger.info(f"Forecast results: preds_df: \n{preds_df} \npreds_df.shape: {preds_df.shape}")
        # 最终预测值保存
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._pred_results_save(preds, preds_df, pred_results_path)
        logger.info(pred_results_path)
        # 预测结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        predict_result_visual(preds_plot, trues_plot, pred_results_path, iters=None)
        logger.info(pred_results_path)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecasting Finished!")
        logger.info(f"{40 * '-'}")
        
        return
