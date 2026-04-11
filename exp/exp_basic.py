# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_basic.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-13
# * Version     : 1.0.021317
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
from importlib import import_module
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Exp_Basic:

    def __init__(self, args):
        # 参数
        self.args = args
        # 模型集 
        # self.non_transformer = [
        #     "DLinear",
        # ]
        self.model_dict = {
            # ------------------------------
            # Time Series Library models
            # ------------------------------
            # 'TimesNet': TimesNet,
            'Autoformer': "models.transformer.Autoformer",
            'Transformer': "models.transformer.Transformer",
            'LSTMTransformer': "models.transformer.LSTMTransformer",
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': "models.mlp.DLinear",
            # 'FEDformer': FEDformer,
            # 'Informer': Informer,
            # 'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            'PatchTST': "models.transformer.PatchTST",
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            # 'Crossformer': Crossformer,
            # 'FiLM': FiLM,
            'iTransformer': "models.transformer.iTransformer",
            # 'Koopa': Koopa,
            # 'TiDE': TiDE,
            # 'FreTS': FreTS,
            # 'MambaSimple': MambaSimple,
            'TimeKAN': "models.others.TimeKAN",
            'TimeMixer': "models.others.TimeMixer",
            'TSMixer': "models.mlp.TSMixer",
            'N_HiTs': "models.mlp.N_HiTs",
            'N_BEATS': "models.mlp.N_BEATS",
            # 'SegRNN': SegRNN,
            # 'TemporalFusionTransformer': TemporalFusionTransformer,
            # "SCINet": SCINet,
            # 'PAttn': PAttn,
            # 'TimeXer': TimeXer,
            # 'WPMixer': WPMixer,
            # 'MultiPatchFormer': MultiPatchFormer
            # ------------------------------
            # Basic Neural Network model
            # ------------------------------
            # "MLP": MLP,
            "RNN": "models.rnn.RNN",
            "GRU": "models.rnn.GRU",
            # "LSTM": LSTM,
            # "BiLSTM": BiLSTM,
            # "Attention": Attention,
            # "CNN_Attention": CNN_Attention,
            # "CNN_Conv1D": CNN_Conv1D,
            # "CNN_Conv2D": CNN_Conv2D,
            # "CNN_LSTM_Attention": CNN_LSTM_Attention,
            # "InformerTodo": InformerTodo, 
            # "Seq2Seq_LSTM": Seq2Seq_LSTM,
            # "LSTM_Attention": LSTM_Attention,
            # "LSTM_CNN": LSTM_CNN,
            # "TCN": TCN,
            # "Transformer": Transformer,
            "LSTM2LSTM": "models.rnn.LSTM2LSTM",
        }
        if args.model == 'Mamba':
            logger.info('Please make sure you have successfully installed mamba_ssm')
            self.model_dict["Mamba"] = "models.rnn.Mamba"
        # 设备
        self.device = self._acquire_device()
        # 模型构建
        self.model = self._build_model().to(self.device)

    def get_model_module(self, model_name: str):
        return import_module(self.model_dict[model_name])
    
    def _acquire_device(self):
        # use gpu or not
        self.args.use_gpu = True \
            if self.args.use_gpu and (torch.cuda.is_available() or torch.backends.mps.is_available()) \
            else False
        # gpu type: "cuda", "mps"
        self.args.gpu_type = self.args.gpu_type.lower().strip()
        # gpu device ids list
        self.args.devices = self.args.devices.replace(" ", "")
        self.args.device_ids = [int(id_) for id_ in self.args.devices.split(",")]
        # gpu device ids string
        self.gpu = self.args.device_ids[0]  # or self.gpu = "0"
        # device
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f"cuda:{self.gpu}")
            logger.info(f"\t\tUse device GPU: cuda:{self.gpu}")
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps") \
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
                else torch.device("cpu")
            logger.info(f"\t\tUse device GPU: mps")
        else:
            device = torch.device("cpu")
            logger.info("\t\tUse device CPU")

        return device
 
    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def valid(self):
        pass
    
    def train(self):
        pass

    def test(self):
        pass
    
    def forecast(self):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
