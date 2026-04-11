import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path: sys.path.append(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic_ltsfm import Exp_Basic
from data_provider.TFs_type.data_factory import data_provider
from utils.metrics_dl import metric
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.plot_results import predict_result_visual
from utils.model_memory import model_memory_size
from utils.log_util import logger

import warnings
warnings.filterwarnings('ignore')


class Exp_Zero_Shot_Forecast(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Exp_Zero_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        """
        模型构建
        """
        # 时间序列模型初始化
        logger.info(f"Initializing model {self.args.model}...")
        model = self.model_dict[self.args.model](self.args).float()
        # 多 GPU 训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # 打印模型参数量
        model_memory_size(model, verbose=True)
        
        return model

    def _get_data(self, flag):
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

    def _test_results_save(self, preds, trues, setting, path,
                           stitched_preds=None,
                           stitched_trues=None,
                           overlap_counts=None,
                           stitched_dates=None):
        """
        测试结果保存
        """
        # ------------------------------
        # 计算窗口级测试结果评价指标
        # ------------------------------
        # 窗口级测试结果
        (window_r2, window_mse, window_rmse, window_mae, window_mape, window_mape_accuracy, window_mspe, window_dtw) = metric(
            preds, trues, 
            use_dtw=self.args.use_dtw
        )
        window_summary_line = (
            f"Window metrics: r2:{window_r2:.4f}, mse:{window_mse:.4f}, rmse:{window_rmse:.4f}, "
            f"mae:{window_mae:.4f}, mape:{window_mape:.4f}, mape accuracy:{window_mape_accuracy:.4f}, "
            f"mspe:{window_mspe:.4f}"
        )
        logger.info(window_summary_line)
        # 缝合的级测试结果
        stitched_summary_line = None
        if stitched_preds is not None and stitched_trues is not None:
            (stitched_r2, stitched_mse, stitched_rmse, stitched_mae, stitched_mape, stitched_mape_accuracy, stitched_mspe, stitched_dtw) = metric(
                stitched_preds.reshape(-1, 1),
                stitched_trues.reshape(-1, 1),
                use_dtw=self.args.use_dtw
            )
            stitched_summary_line = (
                f"Stitched metrics: r2:{stitched_r2:.4f}, mse:{stitched_mse:.4f}, rmse:{stitched_rmse:.4f}, "
                f"mae:{stitched_mae:.4f}, mape:{stitched_mape:.4f}, mape accuracy:{stitched_mape_accuracy:.4f}, "
                f"mspe:{stitched_mspe:.4f}"
            )
            logger.info(stitched_summary_line)

        with open(Path(path).joinpath("result_forecast.txt"), 'w', encoding='utf-8') as file:
            file.write(setting + "  \n")
            file.write(window_summary_line)
            file.write('\n')
            if stitched_summary_line is not None:
                file.write(stitched_summary_line)
            file.write('\n')
            file.write('\n')
            file.close()
        # ------------------------------
        # 测试集上的预测值、真实值 
        # ------------------------------
        # 无缝合的测试集上的预测值、真实值
        flat_results = pd.DataFrame({
            "preds": preds.reshape(-1),
            "trues": trues.reshape(-1),
        })
        flat_results.to_csv(Path(path).joinpath("test_results_windows.csv"), index=False, encoding="utf-8")
        # 缝合的测试集上的预测值、真实值
        if stitched_preds is not None and stitched_trues is not None:
            test_results = self._build_stitched_results_frame(stitched_preds, stitched_trues, overlap_counts, stitched_dates)
        else:
            test_results = flat_results.copy()
            test_results.insert(0, "step", np.arange(len(test_results)))
        test_results.to_csv(Path(path).joinpath("test_results.csv"), index=False, encoding="utf-8")
        logger.info(f"test_results: \n{test_results.head()}")
    
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

    @staticmethod
    def _reshape_target_column(values: np.ndarray, target_idx: int) -> np.ndarray:
        return values[:, :, target_idx:target_idx + 1]
    
    def test(self, setting, test=0):
        """
        模型测试
        """
        # 数据集构建
        test_data, test_loader = self._get_data(flag='test')
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
                # 数据预处理
                # ---------------------
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                # ---------------------
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # ---------------------
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # pred and true process
                # ---------------------
                # pred and true 提取
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                # output detach device
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # 输入输出逆转换
                if test_data.scale and self.args.inverse:
                    if self.args.features == 'MS':
                        outputs = test_data.inverse_transform_target(outputs)
                        batch_y = test_data.inverse_transform_target(
                            self._reshape_target_column(batch_y, test_data.target_idx)
                        )
                    else:
                        outputs = test_data.inverse_transform_full(outputs)
                        batch_y = test_data.inverse_transform_full(batch_y)
                # 预测值/真实值提取
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                # 测试结果收集
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)

                # 预测数据可视化
                # if iters % 100 == 0:
                #     inputs = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         inputs = test_data.inverse_transform_history(inputs)
                #     true_plot = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                #     pred_plot = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                #     predict_result_visual(pred_plot, true_plot, test_results_path, iters=iters)
        # 测试结果处理
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        logger.info(f'test preds shape: {preds.shape}, trues shape: {trues.shape}')
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        logger.info(f'test preds shape: {preds.shape} tures shape: {trues.shape}')
        
        stitched_preds, stitched_trues, overlap_counts = self._stitch_window_predictions(preds, trues)
        stitched_dates = self._build_test_stitched_dates(test_data, len(stitched_preds))
        
        # 测试结果收集
        logger.info(f"{40 * '-'}")
        logger.info(f"Test metric results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._test_results_save(
            preds.reshape(-1, 1),
            trues.reshape(-1, 1),
            setting,
            test_results_path,
            stitched_preds=stitched_preds,
            stitched_trues=stitched_trues,
            overlap_counts=overlap_counts,
            stitched_dates=stitched_dates,
        )
        logger.info(test_results_path)
        
        # 测试结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Test visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        target_dim = -1 if self.args.features in ['M', 'MS'] else 0
        preds_flat = stitched_preds[:, target_dim]
        trues_flat = stitched_trues[:, target_dim]
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

    def forecast(self, setting):
        """
        模型预测（推理）
        """
        pred_data, pred_loader = self._get_data(flag='pred')
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(pred_loader))
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        dec_zeros = torch.zeros((batch_x.shape[0], self.args.pred_len, batch_x.shape[-1]), dtype=batch_x.dtype, device=self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1).float().to(self.device)
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
        # 模型评估模式
        self.model.eval()
        with torch.no_grad():
            if self.args.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 预测结果提取
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, :]
        outputs = outputs[:, :, f_dim:]
        preds = outputs.detach().cpu().numpy()[0]
        history_values = getattr(pred_data, "scaled_history_values", batch_x.detach().cpu().numpy()[0])
        feature_names = getattr(pred_data, "feature_names", [self.args.target])
        history_dates = pd.to_datetime(getattr(pred_data, "history_dates", np.arange(history_values.shape[0])))
        future_dates = pd.to_datetime(getattr(pred_data, "future_dates", np.arange(preds.shape[0])))
        pred_columns = getattr(pred_data, "pred_columns", feature_names[f_dim:] if f_dim != 0 else feature_names)

        if pred_data.scale and self.args.inverse:
            history_values = getattr(pred_data, "raw_history_values", pred_data.inverse_transform_history(history_values))
            if self.args.features == 'MS':
                preds = pred_data.inverse_transform_target(preds)
            else:
                preds = pred_data.inverse_transform_full(preds)
        
        # 预测结果保存
        if len(pred_columns) != preds.shape[-1]:
            pred_columns = pred_columns[-preds.shape[-1]:]
        # 历史数据表
        history_frame = pd.DataFrame(history_values, columns=feature_names)
        history_frame.insert(0, 'date', history_dates)
        # 预测数据表
        forecast_frame = pd.DataFrame(preds, columns=pred_columns)
        forecast_frame.insert(0, 'date', future_dates)
        # 最终预测值保存
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._pred_results_save(history_frame, forecast_frame,  preds, pred_results_path, setting)
        logger.info(pred_results_path)
        # 预测结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        history_target = history_frame[pred_columns[-1]].to_numpy()
        forecast_target = forecast_frame[pred_columns[-1]].to_numpy()
        forecast_target = np.concatenate((history_target, forecast_target), axis=0)
        predict_result_visual(forecast_target, history_target, pred_results_path, iters=None)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecasting Finished!")
        logger.info(f"{40 * '-'}")
        
        return
