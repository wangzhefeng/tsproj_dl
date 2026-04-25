import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
# data pipeline
from data_provider.RNNs_type.data_factory import data_provider
# model training
from utils.model_tools import adjust_learning_rate, EarlyStopping
# loss
from utils.losses import mape_loss, mase_loss, smape_loss
# metrics
from utils.metrics_dl import metric
from utils.plot_results import predict_result_visual
from utils.plot_losses import plot_losses
# log
from utils.model_memory import model_memory_size
from utils.timestamp_utils import from_unix_time
from utils.log_util import logger


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
        model = self.get_model_module(self.args.model).Model(self.args)
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
        raise ValueError(f"Unsupported loss: {self.args.loss}")
    
    def _select_optimizer(self):
        """
        优化器
        """
        if self.args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr = self.args.learning_rate,
            )
        elif self.args.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr = self.args.learning_rate,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        
        return optimizer
    
    @staticmethod
    def _align_prediction_target(outputs, targets):
        """
        Align RNN outputs and labels before loss calculation.
        """
        if outputs.ndim == 2 and targets.ndim == 3 and targets.shape[-1] == 1:
            targets = targets.reshape(outputs.shape)
        if outputs.shape == targets.shape:
            return outputs, targets
        if outputs.ndim == 3 and targets.ndim == 2 and outputs.shape[-1] == 1:
            targets = targets.unsqueeze(-1)
        if outputs.shape == targets.shape:
            return outputs, targets
        raise ValueError(f"Prediction/target shape mismatch: outputs={tuple(outputs.shape)}, targets={tuple(targets.shape)}")
    
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

    def _get_scaler_path(self, setting):
        """
        数据转换器保存路径
        """
        scaler_path = Path(self.args.checkpoints).joinpath(setting, "scalers.pkl")

        return scaler_path

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
                           stitched_dates=None,
                           target_dim=None):
        """
        测试结果保存
        """
        # ------------------------------
        # 计算窗口级测试结果评价指标
        # ------------------------------
        # 窗口级测试结果
        window_r2, window_mse, window_rmse, window_mae, window_mape, window_mape_accuracy, window_mspe, window_dtw = metric(
            preds.reshape(-1, preds.shape[-1]),
            trues.reshape(-1, trues.shape[-1]),
            use_dtw=self.args.use_dtw,
        )
        window_summary_line = (
            f"Window metrics: r2:{window_r2:.4f}, mse:{window_mse:.4f}, rmse:{window_rmse:.4f}, "
            f"mae:{window_mae:.4f}, mape:{window_mape:.4f}, mape accuracy:{window_mape_accuracy:.4f}, "
            f"mspe:{window_mspe:.4f}, dtw:{window_dtw}"
        )
        logger.info(window_summary_line)
        # 真实时间轴级测试结果
        timeline_summary_line = None
        if stitched_preds is not None and stitched_trues is not None:
            (timeline_r2, timeline_mse, timeline_rmse, timeline_mae, timeline_mape, timeline_mape_accuracy, timeline_mspe, timeline_dtw) = metric(
                stitched_preds.reshape(-1, stitched_preds.shape[-1]),
                stitched_trues.reshape(-1, stitched_trues.shape[-1]),
                use_dtw=self.args.use_dtw,
            )
            timeline_summary_line = (
                f"Timeline metrics: r2:{timeline_r2:.4f}, mse:{timeline_mse:.4f}, rmse:{timeline_rmse:.4f}, "
                f"mae:{timeline_mae:.4f}, mape:{timeline_mape:.4f}, "
                f"mape accuracy:{timeline_mape_accuracy:.4f}, mspe:{timeline_mspe:.4f}, dtw:{timeline_dtw}"
            )
            logger.info(timeline_summary_line)

        target_summary_line = None
        if target_dim is not None and preds.shape[-1] > 1:
            target_idx = target_dim if target_dim >= 0 else preds.shape[-1] + target_dim
            target_preds = preds[..., target_idx:target_idx + 1].reshape(-1, 1)
            target_trues = trues[..., target_idx:target_idx + 1].reshape(-1, 1)
            target_r2, target_mse, target_rmse, target_mae, target_mape, target_mape_accuracy, target_mspe, target_dtw = metric(
                target_preds,
                target_trues,
                use_dtw=self.args.use_dtw,
            )
            target_summary_line = (
                f"Target metrics: r2:{target_r2:.4f}, mse:{target_mse:.4f}, rmse:{target_rmse:.4f}, "
                f"mae:{target_mae:.4f}, mape:{target_mape:.4f}, "
                f"mape accuracy:{target_mape_accuracy:.4f}, mspe:{target_mspe:.4f}, dtw:{target_dtw}"
            )
            logger.info(target_summary_line)

        with open(Path(path).joinpath("result_forecast.txt"), "w", encoding="utf-8") as file:
            file.write(setting + "  \n")
            file.write(window_summary_line)
            file.write('\n')
            if timeline_summary_line is not None:
                file.write(timeline_summary_line)
                file.write('\n')
            if target_summary_line is not None:
                file.write(target_summary_line)
                file.write('\n')
            file.write('\n')
            file.close()
        # ------------------------------
        # 测试集上的预测值、真实值
        # ------------------------------
        # 无缝合的测试集上的预测值、真实值
        flat_results = pd.DataFrame({"preds": preds.reshape(-1), "trues": trues.reshape(-1)})
        flat_results.to_csv(Path(path).joinpath("test_results_windows.csv"), index=False, encoding="utf-8")
        # 缝合的测试集上的预测值、真实值
        if stitched_preds is not None and stitched_trues is not None:
            test_results = self._build_stitched_results_frame(stitched_preds, stitched_trues, overlap_counts, stitched_dates)
        else:
            test_results = flat_results.copy()
            test_results.insert(0, "step", np.arange(len(test_results)))
        test_results.to_csv(Path(path).joinpath("test_results.csv"), index=False, encoding="utf-8")
        logger.info(f"test_results: \n{test_results.head()}")
        np.save(path.joinpath('metrics.npy'), np.array([
            window_r2, window_mae, window_mse, window_rmse, window_mape, window_mape_accuracy, window_mspe, window_dtw
        ], dtype=object))
        np.save(path.joinpath('preds.npy'), preds)
        np.save(path.joinpath('trues.npy'), trues)
    
    @staticmethod
    def _git_revision():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return "unknown"

    @staticmethod
    def _require_file(path, description):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"required {description} not found: {path}")
        return path

    def _build_forecast_metadata(
        self,
        setting,
        pred_data,
        history_frame,
        forecast_frame,
        checkpoint_path,
        scaler_path,
        scaler_loaded_from_artifact,
        elapsed_seconds,
    ):
        return {
            "setting": setting,
            "model": getattr(self.args, "model", None),
            "model_id": getattr(self.args, "model_id", None),
            "task_name": getattr(self.args, "task_name", None),
            "features": getattr(self.args, "features", None),
            "target": getattr(self.args, "target", None),
            "feature_names": getattr(pred_data, "feature_names", []),
            "prediction_columns": list(forecast_frame.columns[1:]),
            "seq_len": getattr(self.args, "seq_len", None),
            "pred_len": getattr(self.args, "pred_len", None),
            "pred_method": getattr(self.args, "pred_method", None),
            "freq": getattr(self.args, "freq", None),
            "scale": bool(getattr(self.args, "scale", 0)),
            "inverse": bool(getattr(self.args, "inverse", 0)),
            "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
            "scaler_path": str(Path(scaler_path).resolve()) if scaler_path else None,
            "scaler_loaded_from_artifact": bool(scaler_loaded_from_artifact),
            "history_start": str(history_frame["date"].iloc[0]),
            "history_end": str(history_frame["date"].iloc[-1]),
            "forecast_start": str(forecast_frame["date"].iloc[0]),
            "forecast_end": str(forecast_frame["date"].iloc[-1]),
            "history_points": len(history_frame),
            "forecast_points": len(forecast_frame),
            "device": str(self.device),
            "elapsed_seconds": float(elapsed_seconds),
            "git_commit": self._git_revision(),
        }

    def _pred_results_save(self, trues_df, preds_df, preds=None, path="./", setting=None, metadata=None):
        """
        预测结果保存
        """
        path = Path(path)
        if preds is not None:
            np.save(path.joinpath("prediction.npy"), preds) 
        if trues_df is not None:
            trues_df.to_csv(path.joinpath("history.csv"), index=False, encoding="utf_8_sig")
        if preds_df is not None:
            preds_df.to_csv(path.joinpath("forecast.csv"), index=False, encoding="utf_8_sig")
        if metadata is None:
            metadata = {}
        with open(path.joinpath("metadata.json"), "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        with open(path.joinpath("summary.txt"), "w", encoding="utf-8") as summary_file:
            summary_file.write((setting or "") + "\n")
            summary_file.write(f"prediction only: no ground truth available\n")
            summary_file.write(f"history_points:{len(trues_df)}, forecast_points:{len(preds_df)}\n")
            summary_file.write(f"forecast_target:{preds_df.columns[-1]}\n")
            if metadata:
                summary_file.write(f"checkpoint_path:{metadata.get('checkpoint_path')}\n")
                summary_file.write(f"scaler_path:{metadata.get('scaler_path')}\n")
                summary_file.write(f"scaler_loaded_from_artifact:{metadata.get('scaler_loaded_from_artifact')}\n")
                summary_file.write(f"history_range:{metadata.get('history_start')} -> {metadata.get('history_end')}\n")
                summary_file.write(f"forecast_range:{metadata.get('forecast_start')} -> {metadata.get('forecast_end')}\n")
                summary_file.write(f"device:{metadata.get('device')}\n")
    
    @staticmethod
    def _select_target_column(data, values):
        if values.shape[-1] == 1:
            return values
        target_idx = getattr(data, "target_idx", values.shape[-1] - 1)
        return values[..., target_idx:target_idx + 1]

    def _model_forward(self, data, batch_x, batch_y=None, flag="train", reverse=False):
        """
        RNN 前向传播与输出/目标统一处理。
        """
        # 数据预处理
        # ---------------------
        batch_x = batch_x.float().to(self.device)
        if batch_y is not None:
            batch_y = batch_y.float().to(self.device)

        pred_method = getattr(self.args, "pred_method", "direct_multi_step")
        if pred_method == "seq2seq_multi_step":
            raise NotImplementedError("seq2seq_multi_step forecast is not implemented for RNN todo models yet.")

        def _run_model():
            if flag == "pred" and pred_method == "recursive_multi_step":
                return self._forecast_recursive_tensor(batch_x)
            return self.model(batch_x)

        if self.args.use_amp and self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = _run_model()
        else:
            outputs = _run_model()

        if outputs.ndim == 2:
            outputs = outputs.unsqueeze(-1)
        outputs = outputs[:, -self.args.pred_len:, :]
        if batch_y is not None:
            batch_y = batch_y[:, -self.args.pred_len:, :]
            outputs, batch_y = self._align_prediction_target(outputs, batch_y)

        if self.args.features == "MS":
            outputs = self._select_target_column(data, outputs)
            if batch_y is not None:
                batch_y = self._select_target_column(data, batch_y)

        if flag in ["test", "pred"]:
            outputs = outputs.detach().cpu().numpy()
            if batch_y is not None:
                batch_y = batch_y.detach().cpu().numpy()
            if data.scale and reverse:
                outputs = data.inverse_transform(outputs)
                if batch_y is not None:
                    batch_y = data.inverse_transform(batch_y)
        return outputs, batch_y

    def _forecast_recursive_tensor(self, batch_x):
        current = batch_x.clone()
        preds = []
        for _ in range(self.args.pred_len):
            step_output = self.model(current)
            if step_output.ndim == 2:
                step_output = step_output.unsqueeze(1)
            step_pred = step_output[:, -1:, :]
            preds.append(step_pred)

            next_row = current[:, -1:, :].clone()
            if self.args.features in ["S", "MS"]:
                next_row[:, :, -1:] = step_pred[:, :, -1:]
            else:
                next_row = step_pred
            current = torch.cat([current[:, 1:, :], next_row], dim=1)
        return torch.cat(preds, dim=1)
    
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
        train_data.save_scalers(model_checkpoint_path.parent)
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
        logger.info(f"Train steps: {train_steps}")
        # 模型优化器
        optimizer = self._select_optimizer()
        logger.info(f"Train optimizer has builded...")
        # 模型损失函数
        criterion = self._select_criterion()
        logger.info(f"Train criterion has builded...")
        # 早停类实例
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        logger.info(f"Train early stopping instance has builded, patience: {self.args.patience}")
        # 自动混合精度训练
        if self.args.use_amp:
            scaler = torch.amp.GradScaler()
        # 训练、验证结果收集
        train_losses, vali_losses = [], []
        # 分 epoch 训练
        for epoch in range(self.args.train_epochs):
            # time: epoch 训练开始时间
            epoch_start_time = time.time()
            logger.info(f"Epoch: {epoch+1}, \tstart time: {from_unix_time(epoch_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            # epoch 训练结果收集
            iter_count = 0
            train_loss = []
            # 模型训练模式
            self.model.train()
            for i, data_batch in enumerate(train_loader):
                # 当前 epoch 的迭代次数记录
                iter_count += 1
                # 模型优化器梯度归零
                optimizer.zero_grad()
                # 前向传播
                x_train, y_train = data_batch
                outputs, y_train = self._model_forward(
                    train_data,
                    x_train, y_train, 
                    flag="train", reverse=False,
                )
                # 计算训练损失
                loss = criterion(outputs, y_train)
                train_loss.append(loss.item())
                # 当前 epoch-batch 下每 100 个 batch 的训练速度、误差损失
                if (i + 1) % 10 == 0:
                    # 训练速度和时间
                    speed = (time.time() - train_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info(f'Epoch: {epoch + 1}, \tBatch: {i + 1} | train loss: {loss.item():.7f}, \tSpeed: {speed:.4f}s/batch; left time: {left_time:.4f}s')
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
                scheduler=None,
                model_path=model_checkpoint_path,
            )
            if early_stopping.early_stop:
                logger.info(f"Epoch: {epoch + 1}, \tEarly stopping...")
                break
            # 学习率调整
            adjust_learning_rate(optimizer, None, epoch + 1, self.args)
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
            for i, data_batch in enumerate(vali_loader):
                # logger.info(f"Vali step: {i} running...")
                x_vali, y_vali = data_batch
                outputs, y_vali = self._model_forward(
                    vali_data, 
                    x_vali, y_vali, 
                    flag="valid", reverse=False
                )
                # 计算/保存验证损失
                loss = criterion(outputs, y_vali)
                vali_loss.append(loss.item())
        # 计算验证集上所有 batch 的平均验证损失
        vali_loss = np.average(vali_loss)
        logger.info(f"debug::vali_loss: {vali_loss}")
        # 计算模型输出
        self.model.train()
        # log
        logger.info(f"Validating Finished!")
        return vali_loss

    def test(self, setting, load: bool=False):
        """
        模型测试
        """
        model_checkpoint_path = self._get_model_path(setting)
        scaler_path = self._get_scaler_path(setting)
        self.args.scaler_path = str(scaler_path)
        if load:
            self._require_file(model_checkpoint_path, "test checkpoint")
            if getattr(self.args, "scale", 0):
                self._require_file(scaler_path, "test scaler artifact")
        self.args.require_scaler_artifact_for_pred = bool(load and getattr(self.args, "scale", 0))
        # 数据集构建
        test_data, test_loader = self._get_data(flag="test")
        # 模型加载
        if load:
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
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
        # 模型测试次数
        test_steps = len(test_loader)
        logger.info(f"Test total steps: {test_steps}")
        # 模型评估模式
        self.model.eval()
        # 测试结果收集
        preds, trues = [], []
        with torch.no_grad():
            for i, data_batch in enumerate(test_loader):
                if i == 0 or i == test_steps - 1 or i % 50 == 0:
                    logger.info(f"Test step: {i} running...")
                x_test, y_test = data_batch
                outputs, y_test = self._model_forward(
                    test_data,
                    x_test, y_test,
                    flag="test", reverse=bool(self.args.inverse),
                )
                preds.append(outputs)
                trues.append(y_test)
                if i % 100 == 0:
                    inputs = x_test.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        inputs = test_data.inverse_transform(inputs)
                    true_plot = np.concatenate((inputs[0, :, -1], y_test[0, :, -1]), axis=0)
                    pred_plot = np.concatenate((inputs[0, :, -1], outputs[0, :, -1]), axis=0)
                    predict_result_visual(pred_plot, true_plot, path=test_results_path, iters=i)
        # 测试结果处理
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        logger.info(f"Test results: preds.shape: {preds.shape}, trues.shape: {trues.shape}")
        testing_step = getattr(test_data, "testing_step", getattr(self.args, "testing_step", 1))
        stitched_preds, stitched_trues, overlap_counts = self._stitch_window_predictions(preds, trues, step=testing_step)
        stitched_dates = self._build_test_stitched_dates(test_data, len(stitched_preds), step=testing_step)
        # 测试结果收集
        logger.info(f"{40 * '-'}")
        logger.info(f"Test metric results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._test_results_save(
            preds,
            trues,
            setting,
            test_results_path,
            stitched_preds=stitched_preds,
            stitched_trues=stitched_trues,
            overlap_counts=overlap_counts,
            stitched_dates=stitched_dates,
            target_dim=-1 if self.args.features in ["M", "MS"] else 0,
        )
        logger.info(test_results_path)
        # 测试结果可视化
        logger.info(f"{40 * '-'}")
        logger.info(f"Test visual results have been saved in path:")
        logger.info(f"{40 * '-'}")
        target_dim = -1 if self.args.features in ["M", "MS"] else 0
        preds_flat = stitched_preds[:, target_dim]
        trues_flat = stitched_trues[:, target_dim]
        predict_result_visual(preds_flat, trues_flat, path=Path(test_results_path)) 
        logger.info(test_results_path)
        # log
        logger.info(f"{40 * '-'}")
        logger.info(f"Testing Finished!")
        logger.info(f"{40 * '-'}")

        return
    
    @staticmethod
    def _stitch_window_predictions(preds: np.ndarray, trues: np.ndarray, step: int = 1):
        """
        将窗口级预测按测试步长缝合回真实时间轴，重叠位置取均值。
        """
        step = step if step and step > 0 else 1
        num_windows, pred_len, channels = preds.shape
        stitched_len = (num_windows - 1) * step + pred_len
        
        pred_sum = np.zeros((stitched_len, channels), dtype=np.float64)
        true_sum = np.zeros((stitched_len, channels), dtype=np.float64)
        counts = np.zeros((stitched_len, 1), dtype=np.int64)

        for window_idx in range(num_windows):
            start = window_idx * step
            end = start + pred_len
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

    @staticmethod
    def _build_test_stitched_dates(test_data, stitched_len: int, step: int = 1):
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
        """
        forecast_start_time = time.time()
        model_checkpoint_path = self._get_model_path(setting)
        scaler_path = self._get_scaler_path(setting)
        require_artifacts = bool(getattr(self.args, "forecast_require_artifacts", 1))
        if load and require_artifacts:
            self._require_file(model_checkpoint_path, "forecast checkpoint")
            if getattr(self.args, "scale", 0):
                self._require_file(scaler_path, "forecast scaler artifact")
        self.args.scaler_path = str(scaler_path)
        self.args.require_scaler_artifact_for_pred = bool(load and require_artifacts and getattr(self.args, "scale", 0))
        # 构建预测数据集
        pred_data, pred_loader = self._get_data(flag="pred")
        if len(pred_loader) != 1:
            raise ValueError(f"forecast expects exactly one prediction batch, got {len(pred_loader)}")
        # 数据预处理
        batch_x, batch_y = next(iter(pred_loader))
        if load:
            logger.info(f"{40 * '-'}")
            logger.info("Pretrained model has loaded from:")
            logger.info(f"{40 * '-'}")
            self._require_file(model_checkpoint_path, "forecast checkpoint")
            self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device)["model"])
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
        self.model.eval()
        # 模型预测
        with torch.no_grad():
            preds, _ = self._model_forward(
                pred_data,
                batch_x, batch_y,
                flag="pred", reverse=bool(self.args.inverse),
            )
        # 预测结果提取
        preds = preds[0]
        history_values = getattr(pred_data, "scaled_history_values", batch_x.detach().cpu().numpy()[0])
        feature_names = getattr(pred_data, "feature_names", [self.args.target])
        history_dates = pd.to_datetime(getattr(pred_data, "history_dates", np.arange(history_values.shape[0])))
        future_dates = pd.to_datetime(getattr(pred_data, "future_dates", np.arange(preds.shape[0])))
        pred_columns = getattr(pred_data, "pred_columns", [self.args.target] if self.args.features in ["S", "MS"] else feature_names)

        if pred_data.scale and self.args.inverse:
            history_values = getattr(pred_data, "raw_history_values", pred_data.inverse_transform(history_values))
        if len(pred_columns) != preds.shape[-1]:
            pred_columns = pred_columns[-preds.shape[-1]:]
        expected_channels = 1 if self.args.features in ["S", "MS"] else len(feature_names)
        if preds.shape[-1] != expected_channels:
            raise ValueError(
                f"forecast output channel mismatch for features='{self.args.features}': "
                f"expected {expected_channels}, got {preds.shape[-1]}"
            )
        # 历史数据表
        history_frame = pd.DataFrame(history_values, columns=feature_names)
        history_frame.insert(0, "date", history_dates)
        # 预测数据表
        forecast_frame = pd.DataFrame(preds, columns=pred_columns)
        forecast_frame.insert(0, "date", future_dates)
        elapsed_seconds = time.time() - forecast_start_time
        metadata = self._build_forecast_metadata(
            setting=setting,
            pred_data=pred_data,
            history_frame=history_frame,
            forecast_frame=forecast_frame,
            checkpoint_path=model_checkpoint_path if load else None,
            scaler_path=scaler_path if (getattr(self.args, "scale", 0) and load) else None,
            scaler_loaded_from_artifact=bool(getattr(pred_data, "scaler_loaded_from_artifact", False)),
            elapsed_seconds=elapsed_seconds,
        )
        # 最终预测值保存
        logger.info(f"{40 * '-'}")
        logger.info(f"Forecast results have been saved in path:")
        logger.info(f"{40 * '-'}")
        self._pred_results_save(history_frame, forecast_frame, preds, pred_results_path, setting, metadata=metadata)
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
