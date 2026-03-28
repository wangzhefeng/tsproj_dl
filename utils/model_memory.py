# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_memory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-09
# * Version     : 1.0.060914
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
import warnings
warnings.filterwarnings("ignore")

import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


'''
def model_memory_size(model, input_dtype=torch.float32, verbose:bool=True):
    """
    calculate the memory requirements for this model
    """
    # calculate total number of elements per parameter
    total_params = sum([
        param.numel() 
        # param.nelement() 
        for param in model.parameters()
    ])
    # check if gradients are stored for this parameter
    total_grads = sum([
        param.numel() 
        for param in model.parameters() 
        if param.requires_grad
    ])
    # calculate buffer size(non-parameters that require memory)
    total_buffers = sum([
        buffer.numel() 
        for buffer in model.buffers()
    ])
    if verbose:
        # logger.info(f"Model number of parameters: {total_params / 1e6:.2f}M.")
        logger.info(f"Total number of parameters: {(total_params + total_grads + total_buffers) / 1e6:.2f}M.")

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    # convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    if verbose:
        logger.info(f"Model memory used size: {total_memory_gb:.2f}GB.")

    return total_memory_gb
'''


def model_memory_size(model, input_dtype=torch.float32, verbose:bool=True):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size
    
    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())
    
    # Account for weight tying
    total_params_normalized = total_params - model.tok_embed.weight.numel()
    
    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    
    # 返回指定设备上张量当前占用的 GPU 内存字节数, 这很可能小于在 nvidia-smi 中显示的量，
    # 因为一些未使用的内存可能被缓存分配器持有，并且需要在 GPU 上创建一些上下文
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    max_allocated_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    # 返回指定设备上缓存分配器管理的当前 GPU 内存字节数
    cached_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    
    # Logs
    if verbose:
        logger.info(f"{40 * '='}")
        logger.info(f"Model Info:")
        logger.info(f"{40 * '='}")
        logger.info(f"Model number of parameters: {total_params:,}")
        logger.info(f"Model number of parameters: {total_params / 1e6:.2f}M.")
        logger.info(f"Total number of parameters: {(total_params + total_grads + total_buffers):,}")
        logger.info(f"Total number of parameters: {(total_params + total_grads + total_buffers) / 1e6:.2f}M.")
        logger.info(f"Total number of unique parameters: {total_params_normalized:,}")
        logger.info(f"Model memory used size: {total_memory_gb:.2f}GB.")
        logger.info(f"Allocated tensor emory: {allocated_memory:.2f}GB.")
        logger.info(f"Max Allocated tensor emory: {max_allocated_memory:.2f}GB.")
        logger.info(f"Cached total memory: {cached_memory:.2f}GB.")
        logger.info(f"{40 * '='}")




# 测试代码 main 函数
def main():
    dtype_float_64 = torch.float64
    dtype_float_32 = torch.float32
    dtype_float_16 = torch.float16
    dtype_float_default = torch.float
    dtype_int_64 = torch.int64
    dtype_int_32 = torch.int32
    dtype_int_16 = torch.int16
    dtype_int_8 = torch.int8
    dtype_uint_8 = torch.uint8
    dtype_int_default = torch.int
    element_size = torch.tensor(0.0, dtype=dtype_float_64).element_size()
    logger.info(f"torch.float64 element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_float_32).element_size()
    logger.info(f"torch.float32 element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_float_16).element_size()
    logger.info(f"torch.float16 element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_float_default).element_size()
    logger.info(f"torch.float[default] element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_int_64).element_size()
    logger.info(f"torch.int64 element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_int_32).element_size()
    logger.info(f"torch.int32 element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_int_16).element_size()
    logger.info(f"torch.int16 element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_int_8).element_size()
    logger.info(f"torch.int8 element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_uint_8).element_size()
    logger.info(f"torch.uint element size: {element_size}")
    element_size = torch.tensor(0.0, dtype=dtype_int_default).element_size()
    logger.info(f"torch.int[default] element size: {element_size}")

if __name__ == "__main__":
    main()
