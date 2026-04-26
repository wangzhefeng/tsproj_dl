# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_factory_dl_3.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-24
# * Version     : 1.0.052418
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

from torch.utils.data import DataLoader

from data_provider.RNNs_type.data_loader import (
    Dataset_Train, Dataset_Pred,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_provider(args, flag):
    """
    数据集构造
    """
    canonical_flag = flag.lower()
    if canonical_flag not in ["train", "valid", "test", "pred"]:
        raise ValueError(
            f"Unsupported data flag: {flag}. "
            "Expected one of ['train', 'valid', 'test', 'pred']."
        )
    # TODO 是否对时间戳进行编码
    timeenc = 0 if args.embed != "timeF" else 1
    # 仅训练集打乱；验证、测试和预测保持时间顺序，便于复现和结果缝合
    shuffle_flag = True if canonical_flag == "train" else False
    # 是否丢弃最后一个 batch
    drop_last = False
    num_workers = 0 if canonical_flag == "pred" else args.num_workers
    # 数据集参数
    if canonical_flag in ["train", "valid"]:
        batch_size = args.batch_size
        Data = Dataset_Train
    elif canonical_flag == "test":
        batch_size = 1
        Data = Dataset_Train
    elif canonical_flag == "pred":
        batch_size = 1
        Data = Dataset_Pred
    step_attr = {
        "train": "train_step",
        "valid": "valid_step",
        "test": "testing_step",
        "pred": "step_size",
    }[canonical_flag]
    step_size = getattr(args, step_attr, getattr(args, "step_size", 1))
    # 构建 Dataset 和 DataLoader
    data_set = Data(
        args = args,
        root_path = args.root_path,
        data_path = args.data_path,
        flag = canonical_flag,
        features = args.features,
        target = args.target,
        time = args.time,
        freq = args.freq,
        seq_len = args.seq_len,
        pred_len = args.pred_len,
        pred_method = args.pred_method,
        step_size = step_size,
        scale = args.scale,
    )
    logger.info(f"{canonical_flag}: {len(data_set)}")
    data_loader = DataLoader(
        dataset = data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        sampler=None,
        drop_last = drop_last,
        num_workers=num_workers,
    )
    
    return data_set, data_loader




def build_local_test_args(**overrides):
    """
    构造 RNN 数据管道本地 smoke 参数。
    """
    from types import SimpleNamespace
    args = dict(
        root_path="./dataset/ETT-small",
        data_path="ETTh1.csv",
        target="OT",
        time="date",
        freq="h",
        features="MS",
        seq_len=24,
        pred_len=6,
        pred_method="direct_multi_output",
        step_size=1,
        train_step=4,
        valid_step=4,
        testing_step=1,
        batch_size=4,
        train_ratio=0.7,
        test_ratio=0.2,
        embed="timeF",
        scale=1,
        num_workers=0,
        feature_size=7,
        hidden_size=32,
        num_layers=2,
        rolling_predict=1,
        rolling_data_path="ETTh1.csv",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def main():
    """
    本地快速测试 RNN 数据管道。
    """
    from utils.log_util import logger

    args = build_local_test_args()
    for flag in ["train", "valid", "test", "pred"]:
        logger.info(f"{40 * '='}")
        logger.info(f"RNN data provider local smoke: flag={flag}")
        logger.info(f"{40 * '='}")
        data_set, data_loader = data_provider(args, flag=flag)
        batch_x, batch_y = next(iter(data_loader))
        logger.info(
            f"flag={flag}, dataset_len={len(data_set)}, loader_len={len(data_loader)}, "
            f"batch_size={data_loader.batch_size}, shuffle={data_loader.sampler.__class__.__name__}"
        )
        logger.info(
            f"feature_dim={data_set.feature_dim}, target_dim={data_set.target_dim}, "
            f"feature_names={data_set.feature_names}, pred_columns={data_set.pred_columns}"
        )
        logger.info(f"batch_x.shape={tuple(batch_x.shape)}, batch_y.shape={tuple(batch_y.shape)}")

if __name__ == "__main__":
    main()
