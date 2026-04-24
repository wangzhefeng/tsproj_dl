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

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import importlib

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Exp_Basic:

    def __init__(self, args):
        # 参数
        self.args = args
        # 模型集
        model_map = self._scan_models_directory(model_type="cnn")
        model_map.update(self._scan_models_directory(model_type="gnn"))
        model_map.update(self._scan_models_directory(model_type="ltsfm"))
        model_map.update(self._scan_models_directory(model_type="mlp"))
        model_map.update(self._scan_models_directory(model_type="others"))
        model_map.update(self._scan_models_directory(model_type="rnn"))
        model_map.update(self._scan_models_directory(model_type="transformer"))
        self.model_dict = LazyModelDict(model_map)
        # 设备
        self.device = self._acquire_device()
        # 模型构建
        self.model = self._build_model().to(self.device)
    
    def _scan_models_directory(self, model_type: str):
        """
        Automatically scan all .py files in the models folder
        """
        model_map = {}
        models_dir = f"models/{model_type}"
        # Iterate through all files in 'models' directory
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                # Ignore __init__.py and non-.py files
                if filename.endswith('.py') and filename != '__init__.py':
                    # Remove .py extension to get module name
                    module_name = filename[:-3]
                    # Build full import path
                    full_path = f"{models_dir}.{module_name}"
                    # loading dict: {'Transformer': 'models.Transformer'}
                    model_map[module_name] = full_path
        
        return model_map
    
    def get_model_module(self, model_name: str):
        return importlib.import_module(self.model_dict[model_name])
    
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


class LazyModelDict(dict):
    """
    Smart Lazy-Loading Dictionary
    """
    def __init__(self, model_map):
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        
        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")
            
        module_path = self.model_map[key]
        try:
            print(f"🚀 Lazy Loading: {key} ...") 
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # Try to find the model class
        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        self[key] = model_class
        return model_class
