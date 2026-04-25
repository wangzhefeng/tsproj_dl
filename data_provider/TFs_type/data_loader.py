# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-04-17
# * Version     : 1.0.041713
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features
from utils.frequency import resolve_freq_step_minutes
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _build_calendar_features(stamps, time_col, freq, timeenc):
    df_stamp = pd.DataFrame({time_col: pd.to_datetime(stamps)})
    if timeenc == 0:
        freq_num = resolve_freq_step_minutes(freq)
        df_stamp['month'] = df_stamp[time_col].apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp[time_col].apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp[time_col].apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp[time_col].apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp[time_col].apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp[time_col].map(lambda x: x.minute // freq_num)
        return df_stamp.drop([time_col], axis=1).values

    data_stamp = time_features(pd.to_datetime(df_stamp[time_col].values), freq=freq)
    return data_stamp.transpose(1, 0)


def _validate_dataframe_columns(df_raw, time_col, target_col):
    if time_col not in df_raw.columns:
        raise ValueError(f"time column '{time_col}' not found in data")
    if target_col not in df_raw.columns:
        raise ValueError(f"target column '{target_col}' not found in data")


def _reorder_dataframe(df_raw, time_col, target_col):
    cols = list(df_raw.columns)
    cols.remove(target_col)
    cols.remove(time_col)
    return df_raw[[time_col] + cols + [target_col]]


def _ensure_single_target_channel(data):
    if np.asarray(data).shape[-1] != 1:
        raise ValueError("target inverse transform expects data with exactly one target channel")


def _scaler_artifact_path(path):
    path = Path(path)
    if path.suffix:
        return path
    return path.joinpath("scalers.pkl")


def _metadata_artifact_path(path):
    path = Path(path)
    if path.suffix:
        return path.with_name("scaler_metadata.json")
    return path.joinpath("scaler_metadata.json")


def _dump_scaler_artifacts(path, full_scaler, target_scaler, feature_names, target, target_idx, features):
    scaler_path = _scaler_artifact_path(path)
    metadata_path = _metadata_artifact_path(path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as file:
        pickle.dump(
            {
                "full_scaler": full_scaler,
                "target_scaler": target_scaler,
            },
            file,
        )
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_names": feature_names,
                "target": target,
                "target_idx": target_idx,
                "features": features,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return scaler_path


def _load_scaler_artifacts(path):
    scaler_path = _scaler_artifact_path(path)
    metadata_path = _metadata_artifact_path(path)
    with open(scaler_path, "rb") as file:
        scalers = pickle.load(file)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    return scalers, metadata


class Dataset_Train(Dataset):
    
    def __init__(self,
                 args,
                 root_path,
                 data_path,
                 flag='train',
                 size=None,  # size [seq_len, label_len, pred_len]
                 features='MS',
                 target='OT',
                 time="time",
                 freq='15min',
                 timeenc=0,
                 seasonal_patterns=None,
                 scale=True,
                 inverse=False,
                 testing_step=1):
        self.args = args
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        # data type
        normalized_flag = flag.lower()
        assert normalized_flag in ['train', 'test', 'valid']
        self.flag = normalized_flag
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[self.flag]
        # data size
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        # data freq, feature columns, and target
        self.features = features
        self.target = target
        self.time = time
        self.freq = freq
        self.timeenc = timeenc
        self.seasonal_patterns = seasonal_patterns
        # data preprocess
        self.scale = scale
        self.inverse = inverse
        self.train_step = getattr(self.args, "train_step", 1)
        self.valid_step = getattr(self.args, "valid_step", 1)
        self.testing_step = testing_step
        # data read
        self.__read_data__()

    def __read_data__(self):
        logger.info(f"{40 * '-'}")
        logger.info(f"Load and Preprocessing {self.flag} data...")
        logger.info(f"{40 * '-'}")
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        logger.info(f"Train data shape: {df_raw.shape}")
        # 数据变量验证
        _validate_dataframe_columns(df_raw, self.time, self.target)
        # 数据特征排序
        df_raw = _reorder_dataframe(df_raw, self.time, self.target)
        logger.info(f"Train data shape after feature order: {df_raw.shape}")
        # 根据预测任务进行特征筛选
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"unsupported features mode: {self.features}")
        logger.info(f"Train data shape after feature selection: {df_data.shape}")
        self.feature_names = list(df_data.columns)
        self.target_idx = self.feature_names.index(self.target)
        # 数据分割比例
        train_ratio = getattr(self.args, "train_ratio", 0.7)
        test_ratio = getattr(self.args, "test_ratio", 0.2)
        num_train = int(len(df_data) * train_ratio)
        num_test = int(len(df_data) * test_ratio)
        num_vali = len(df_data) - num_train - num_test
        # 数据分割索引
        border1s = [0,         num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali,     len(df_data)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        # 数据标准化
        self.full_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        if self.scale:
            scaler_path = getattr(self.args, "scaler_path", None)
            if self.flag == "test" and scaler_path and _scaler_artifact_path(scaler_path).exists():
                scalers, metadata = _load_scaler_artifacts(scaler_path)
                self.full_scaler = scalers["full_scaler"]
                self.target_scaler = scalers["target_scaler"]
                expected_features = metadata.get("feature_names")
                if expected_features and expected_features != self.feature_names:
                    raise ValueError(
                        f"scaler feature_names mismatch: expected {expected_features}, got {self.feature_names}"
                    )
                expected_target = metadata.get("target")
                if expected_target and expected_target != self.target:
                    raise ValueError(f"scaler target mismatch: expected {expected_target}, got {self.target}")
                logger.info(f"Scaler artifacts have been loaded from path: {_scaler_artifact_path(scaler_path)}")
            else:
                if self.flag == "test" and scaler_path:
                    logger.info(f"Scaler artifacts not found in path: {_scaler_artifact_path(scaler_path)}. Fit scalers with training data.")
                train_data = df_data[border1s[0]:border2s[0]]
                self.full_scaler.fit(train_data.values)
                self.target_scaler.fit(train_data[[self.target]].values)
            data = self.full_scaler.transform(df_data.values)
        else:
            data = df_data.values
        logger.info(f"Train data shape after standardization: {data.shape}")
        # 训练/测试/验证数据集分割: 选取当前 flag 下的数据
        logger.info(f"Train data length: {border2s[0]-border1s[0]}, Valid data length: {border2s[1]-border1s[1]}, Test data length: {border2s[2]-border1s[2]}")
        logger.info(f"Train step: {self.train_step}, Valid step: {self.valid_step}, Test step: {self.testing_step}")
        logger.info(f"{self.flag.capitalize()} input data index: {border1}:{border2}, data length: {border2-border1}")
        # 时间特征处理
        self.segment_dates = pd.to_datetime(df_raw[self.time].iloc[border1:border2]).reset_index(drop=True)
        data_stamp = _build_calendar_features(self.segment_dates, self.time, self.freq, self.timeenc)
        logger.info(f"Train timestamp features shape: {data_stamp.shape}")
        # 数据切分
        self.raw_segment_values = df_data.values[border1:border2].astype(np.float32)
        self.scaled_segment_values = data[border1:border2].astype(np.float32)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # TODO 数据增强
        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
        #         self.data_x, self.data_y, self.args
        #     )
        # logger.info(f"debug::data_x: \n{self.data_x} \ndata_x shape: {self.data_x.shape}")
        # logger.info(f"debug::data_y: \n{self.data_y} \ndata_y shape: {self.data_y.shape}")
        # logger.info(f"debug::data_stamp: \n{self.data_stamp} \ndata_stamp shape: {self.data_stamp.shape}")

    def __getitem__(self, index):
        # data_x 索引
        if self.flag == "train":
            step = self.train_step if self.train_step and self.train_step > 0 else 1
            s_begin = index * step
        elif self.flag == "valid":
            step = self.valid_step if self.valid_step and self.valid_step > 0 else 1
            s_begin = index * step
        elif self.flag == "test":
            step = self.testing_step if self.testing_step and self.testing_step > 0 else 1
            s_begin = index * step
        s_end = s_begin + self.seq_len
        # data_y 索引
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 数据索引分割
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # 时间特征分割
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        total = len(self.data_x) - self.seq_len - self.pred_len + 1
        if total <= 0:
            return 0
        if self.flag == "train":
            step = self.train_step if self.train_step and self.train_step > 0 else 1
        elif self.flag == "valid":
            step = self.valid_step if self.valid_step and self.valid_step > 0 else 1
        else:
            step = self.testing_step if self.testing_step and self.testing_step > 0 else 1
        return (total - 1) // step + 1

    def inverse_transform(self, data):
        return self.inverse_transform_full(data)

    def inverse_transform_full(self, data):
        if not self.scale:
            return data
        original_shape = data.shape
        restored = self.full_scaler.inverse_transform(np.asarray(data).reshape(-1, original_shape[-1]))
        return restored.reshape(original_shape)

    def inverse_transform_target(self, data):
        _ensure_single_target_channel(data)
        if not self.scale:
            return data
        original_shape = data.shape
        restored = self.target_scaler.inverse_transform(np.asarray(data).reshape(-1, 1))
        return restored.reshape(original_shape)

    def inverse_transform_history(self, data):
        return self.inverse_transform_full(data)

    def save_scalers(self, path):
        if not self.scale:
            return None
        scaler_path = _dump_scaler_artifacts(
            path,
            self.full_scaler,
            self.target_scaler,
            self.feature_names,
            self.target,
            self.target_idx,
            self.features,
        )
        logger.info(f"Scaler artifacts have been saved in path: {scaler_path}")
        return scaler_path


class Dataset_Pred(Dataset):
    
    def __init__(self,
                 args,
                 root_path,
                 data_path,
                 flag='pred',
                 size=None,  # size: [seq_len, label_len, pred_len]
                 features='MS',
                 target='OT',
                 time="time",
                 timeenc=0,
                 freq='15min',
                 seasonal_patterns=None,
                 scale=True,
                 inverse=False,
                 testing_step=None):
        self.args = args
        # data file path
        self.root_path = root_path
        self.data_path = data_path
        # data type
        self.flag = flag
        assert flag in ["pred"]
        # data size
        self.seq_len = 24 * 4 * 4 if size is None else size[0]
        self.label_len = 24 * 4 if size is None else size[1]
        self.pred_len = 24 * 4 if size is None else size[2]
        # data freq, feature columns, and target
        self.features = features
        self.target = target
        self.time = time
        self.freq = freq
        self.timeenc = timeenc
        self.seasonal_patterns = seasonal_patterns
        # data preprocess
        self.scale = scale
        self.inverse = inverse
        self.testing_step = testing_step
        # data read
        self.__read_data__()

    def __read_data__(self):
        logger.info(f"{40 * '-'}")
        logger.info(f"Load and Preprocessing {self.flag} data...")
        logger.info(f"{40 * '-'}")
        # 数据文件(CSV)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        logger.info(f"Train data shape: {df_raw.shape}")
        # 数据变量验证
        _validate_dataframe_columns(df_raw, self.time, self.target)
        
        # 数据特征排序
        df_raw = _reorder_dataframe(df_raw, self.time, self.target)
        logger.info(f"Train data shape after feature order: {df_raw.shape}")
        # 预测特征变量数据
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"unsupported features mode: {self.features}")
        logger.info(f"Train data shape after feature selection: {df_data.shape}")
        # 
        self.feature_names = list(df_data.columns)
        self.pred_columns = list(df_data.columns[-1:] if self.features == 'MS' else df_data.columns)
        self.target_idx = self.feature_names.index(self.target)
        if len(df_data) < self.seq_len:
            raise ValueError(f"not enough rows for prediction: need at least seq_len={self.seq_len}, got {len(df_data)}")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be positive in pred mode")
        # 数据转换
        self.full_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        if self.scale:
            scaler_path = getattr(self.args, "scaler_path", None)
            if scaler_path and _scaler_artifact_path(scaler_path).exists():
                scalers, metadata = _load_scaler_artifacts(scaler_path)
                self.full_scaler = scalers["full_scaler"]
                self.target_scaler = scalers["target_scaler"]
                expected_features = metadata.get("feature_names")
                if expected_features and expected_features != self.feature_names:
                    raise ValueError(
                        f"scaler feature_names mismatch: expected {expected_features}, got {self.feature_names}"
                    )
                expected_target = metadata.get("target")
                if expected_target and expected_target != self.target:
                    raise ValueError(f"scaler target mismatch: expected {expected_target}, got {self.target}")
                logger.info(f"Scaler artifacts have been loaded from path: {_scaler_artifact_path(scaler_path)}")
            else:
                if scaler_path:
                    logger.info(f"Scaler artifacts not found in path: {_scaler_artifact_path(scaler_path)}. Fit scalers with current prediction data.")
                self.full_scaler.fit(df_data.values)
                self.target_scaler.fit(df_data[[self.target]].values)
            data = self.full_scaler.transform(df_data.values)
        else:
            data = df_data.values
        logger.info(f"Train data shape after standardization: {data.shape}")
        # 数据窗口索引
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)
        logger.info(f"Forecast input data index: {border1}:{border2}, data length: {border2-border1}")
        # 时间戳特征处理
        forecast_history_stamp = pd.to_datetime(df_raw[self.time].iloc[border1:border2], format='mixed')
        forecast_history_stamp = forecast_history_stamp.reset_index(drop=True)
        forecast_future_stamp = pd.date_range(forecast_history_stamp.iloc[-1], periods=self.pred_len + 1, freq=self.freq)[1:]
        self.forecast_start_time = forecast_future_stamp[0]
        self.history_dates = forecast_history_stamp.to_numpy()
        self.future_dates = forecast_future_stamp.to_numpy()
        combined_stamp = np.concatenate([self.history_dates, self.future_dates], axis=0)
        data_stamp = _build_calendar_features(combined_stamp, self.time, self.freq, self.timeenc)
        logger.info(f"Train and Forecast timestamp features shape: {data_stamp.shape}")
        # 数据切分
        self.raw_history_values = df_data.values[border1:border2].astype(np.float32)
        self.scaled_history_values = data[border1:border2].astype(np.float32)
        self.data_x = self.scaled_history_values
        self.data_y = self.scaled_history_values
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # data_x 索引
        s_begin = index
        s_end = s_begin + self.seq_len
        # data_y 索引
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 数据索引分割
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:(r_begin + self.label_len)]
        # 时间特征分割
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return 1

    def inverse_transform(self, data):
        return self.inverse_transform_full(data)

    def inverse_transform_full(self, data):
        if not self.scale:
            return data
        original_shape = data.shape
        restored = self.full_scaler.inverse_transform(np.asarray(data).reshape(-1, original_shape[-1]))
        return restored.reshape(original_shape)

    def inverse_transform_target(self, data):
        _ensure_single_target_channel(data)
        if not self.scale:
            return data
        original_shape = data.shape
        restored = self.target_scaler.inverse_transform(np.asarray(data).reshape(-1, 1))
        return restored.reshape(original_shape)

    def inverse_transform_history(self, data):
        return self.inverse_transform_full(data)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
