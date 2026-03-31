# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_splitor.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-20
# * Version     : 1.0.012021
# * Description : https://blog.csdn.net/java1314777/article/details/134407174
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "Dataset_Train",
    "Dataset_Pred",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.log_util import logger
from utils.time_col_tools import time_col_distinct, time_col_rename

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class _RNNBaseDataset(Dataset):

    def __init__(self):
        self.scaler = StandardScaler()

    def _validate_common_args(self):
        valid_features = {"M", "S", "MS"}
        valid_pred_methods = {
            "recursive_multi_step",
            "direct_multi_output",
            "direct_multi_step",
            "direct_recursive_multi_step_mix",
        }

        if self.features not in valid_features:
            raise ValueError(
                f"Unsupported features: {self.features}. "
                f"Expected one of {sorted(valid_features)}."
            )
        if self.pred_method not in valid_pred_methods:
            raise ValueError(
                f"Unsupported pred_method: {self.pred_method}. "
                f"Expected one of {sorted(valid_pred_methods)}."
            )
        if self.seq_len <= 0:
            raise ValueError("seq_len must be a positive integer.")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be a positive integer.")
        if self.step_size <= 0:
            raise ValueError("step_size must be a positive integer.")
        if not (0 < self.args.train_ratio <= 1):
            raise ValueError("train_ratio must be in (0, 1].")
        if not (0 <= self.args.test_ratio < 1):
            raise ValueError("test_ratio must be in [0, 1).")
        if (self.args.train_ratio + self.args.test_ratio) > 1:
            raise ValueError("train_ratio + test_ratio must be <= 1.")

    def _read_raw_frame(self, data_path: Path) -> pd.DataFrame:
        df_raw = pd.read_csv(data_path, parse_dates=[self.time])
        logger.info(f"Train data: \n{df_raw.head()}")
        logger.info(f"Train data shape: {df_raw.shape}")
        logger.info(f"Train data NA check: \n{df_raw.isna().sum()}")

        df_raw = time_col_rename(df_raw, time_col=self.time)
        df_raw = time_col_distinct(df_raw, time_col="time")
        logger.info(f"Train data shape after drop timestamp duplicate: {df_raw.shape}")

        df_complete = pd.DataFrame({
            "time": pd.date_range(
                df_raw["time"].min(),
                df_raw["time"].max(),
                freq=self.freq,
            )
        })
        source = df_raw.set_index("time")
        for col in df_raw.columns:
            if col != "time":
                df_complete[col] = df_complete["time"].map(source[col])
        df_raw = df_complete
        logger.info(f"Train data shape after date complete: {df_raw.shape}")

        df_raw.set_index("time", inplace=True)
        df_raw = df_raw.interpolate(method="linear", limit_direction="both")
        df_raw = df_raw.dropna(axis=0)
        df_raw.reset_index(inplace=True)
        logger.info(f"Train data shape after interpolate and dropna: {df_raw.shape}")

        if self.target not in df_raw.columns:
            raise ValueError(f"Target column `{self.target}` not found in {data_path}.")

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("time")
        df_raw = df_raw[["time"] + cols + [self.target]]
        logger.info(f"Train data shape after feature order: {df_raw.shape}")

        return df_raw

    def _select_feature_frame(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if self.features in ["M", "MS"]:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        self.feature_dim = df_data.shape[1]
        self.target_dim = 1 if self.features in ["MS", "S"] else self.feature_dim
        logger.info(f"Train data shape after feature selection: {df_data.shape}")

        return df_data

    def _get_split_borders(self, data_len: int) -> Tuple[List[int], List[int]]:
        num_train = int(data_len * self.args.train_ratio)
        num_test = int(data_len * self.args.test_ratio)
        num_vali = data_len - num_train - num_test

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, data_len]

        return border1s, border2s

    def _fit_scaler(self, df_data: pd.DataFrame):
        border1s, border2s = self._get_split_borders(len(df_data))
        train_data = df_data.iloc[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        return border1s, border2s

    def _transform_frame(self, df_data: pd.DataFrame) -> np.ndarray:
        if self.scale:
            return self.scaler.transform(df_data.values)
        return df_data.values

    def _build_label_window(self, input_data: torch.Tensor, start_idx: int) -> torch.Tensor:
        label_start = start_idx + self.seq_len
        label_end = label_start + self.pred_len

        if self.features in ["MS", "S"]:
            future_target = input_data[label_start:label_end, -1:]
        else:
            future_target = input_data[label_start:label_end]

        return future_target

    def _inverse_input_dim(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return self.scaler.inverse_transform(data)

        shape = data.shape
        return self.scaler.inverse_transform(data.reshape(-1, shape[-1])).reshape(shape)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        data = np.asarray(data)
        if data.shape[-1] == self.feature_dim:
            return self._inverse_input_dim(data)

        target_mean = self.scaler.mean_[-1]
        target_scale = self.scaler.scale_[-1]
        return data * target_scale + target_mean


class Dataset_Train(_RNNBaseDataset):

    def __init__(self,
                 args,
                 root_path: str,
                 data_path: str,
                 target: str,
                 time: str,
                 freq: str,
                 features: str,
                 seq_len: int,
                 pred_len: int,
                 pred_method: str="recursive_multi_step",
                 step_size: int=1,
                 scale: bool=True,
                 flag: str="train"):
        super().__init__()

        self.args = args
        self.data_file_path = Path(root_path).joinpath(data_path)
        self.target = target
        self.time = time
        self.freq = freq
        self.features = features
        self.flag = flag if flag != "val" else "valid"
        assert self.flag in ["train", "test", "valid"]
        type_map = {"train": 0, "valid": 1, "test": 2}
        self.set_type = type_map[self.flag]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pred_method = pred_method
        self.step_size = step_size
        self.scale = bool(scale)

        self._validate_common_args()
        self.__read_data()

    def __read_data(self):
        logger.info(f"{40 * '-'}")
        logger.info(f"Load and Preprocess {self.flag} data...")
        logger.info(f"{40 * '-'}")

        df_raw = self._read_raw_frame(self.data_file_path)
        df_data = self._select_feature_frame(df_raw)

        border1s, border2s = self._fit_scaler(df_data)
        data = self._transform_frame(df_data)
        logger.info(f"Train data shape after standardization: {data.shape}")

        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        data_tensor = torch.as_tensor(data[border1:border2], dtype=torch.float32)
        logger.info(
            f"Train data length: {border2s[0]-border1s[0]}, "
            f"Valid data length: {border2s[1]-border1s[1]}, "
            f"Test data length: {border2s[2]-border1s[2]}"
        )
        logger.info(f"{self.flag.capitalize()} input data index: {border1}:{border2}")
        logger.info(f"{self.flag.capitalize()} input data shape: {data_tensor.shape}")

        self.sequences = self.__create_input_sequences(data_tensor)

    def __create_input_sequences(self, input_data: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        output_seq = []
        input_data_len = len(input_data)

        for i in range(0, input_data_len - self.seq_len, self.step_size):
            if (i + self.seq_len + self.pred_len) > input_data_len:
                break

            train_seq = input_data[i:(i + self.seq_len)]
            train_label = self._build_label_window(input_data, i)
            output_seq.append((train_seq, train_label))

        logger.info(f"{self.flag.capitalize()} sample number: {len(output_seq)}")

        return output_seq

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return sequence, label


class Dataset_Pred(_RNNBaseDataset):

    def __init__(self,
                 args,
                 root_path: str,
                 data_path: str,
                 target: str,
                 time: str,
                 freq: str,
                 features: str,
                 seq_len: int,
                 pred_len: int,
                 pred_method: str="recursive_multi_step",
                 step_size: int=1,
                 scale: bool=True,
                 flag: str="pred"):
        super().__init__()

        self.args = args
        self.data_file_path = Path(root_path).joinpath(data_path)
        self.target = target
        self.time = time
        self.freq = freq
        self.features = features
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pred_method = pred_method
        self.step_size = step_size
        self.scale = bool(scale)
        self.rolling_data_path = getattr(args, "rolling_data_path", data_path)

        self._validate_common_args()
        self.__read_data()

    def __read_data(self):
        logger.info(f"{40 * '-'}")
        logger.info("Load and Preprocess pred data...")
        logger.info(f"{40 * '-'}")

        history_raw = self._read_raw_frame(self.data_file_path)
        history_data = self._select_feature_frame(history_raw)
        self._fit_scaler(history_data)

        rolling_path = Path(self.args.root_path).joinpath(self.rolling_data_path)
        pred_source_path = rolling_path if rolling_path.exists() else self.data_file_path
        pred_raw = self._read_raw_frame(pred_source_path)
        pred_data = self._select_feature_frame(pred_raw)
        pred_array = self._transform_frame(pred_data)

        data_tensor = torch.as_tensor(pred_array, dtype=torch.float32)
        if len(data_tensor) < self.seq_len:
            raise ValueError(
                f"Prediction data length {len(data_tensor)} is smaller than seq_len {self.seq_len}."
            )

        seq_x = data_tensor[-self.seq_len:]
        if len(data_tensor) >= (self.seq_len + self.pred_len):
            seq_y = self._build_label_window(data_tensor, len(data_tensor) - self.seq_len - self.pred_len)
        else:
            seq_y = torch.zeros(self.pred_len, self.target_dim, dtype=torch.float32)

        self.sequences = [(seq_x, seq_y)]
        logger.info(f"Pred input data shape: {seq_x.shape}")
        logger.info(f"Pred label placeholder shape: {seq_y.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return sequence, label


# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()
