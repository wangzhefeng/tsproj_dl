# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LSTM.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-27
# * Version     : 0.1.052722
# * Description : description
# * Link        : https://weibaohang.blog.csdn.net/article/details/128605806
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Model_v1(nn.Module):
    
    def __init__(self, args):
        super(Model_v1, self).__init__()
        
        self.args = args
        # lstm
        self.lstm = nn.LSTM(
            input_size = args.feature_size, 
            hidden_size = args.hidden_size, 
            num_layers = args.num_layers, 
            bias = True,
            batch_first = True
        )
        # output size
        output_size = 1 if (args.features == "MS" or args.features == "S") else args.feature_size
        # fc layer
        self.linear = nn.Linear(
            in_features = args.hidden_size, 
            out_features = output_size,
        )

    def forward(self, x, hidden = None):
        # 获取批次大小
        batch_size, seq_len, feature_size = x.shape[0]  # x.shape=(batch_size, seq_len, feature_size)
        
        # 初始化隐藏状态
        if hidden is None:
            # (D*num_layers, batch_size, hidden_size)
            h_0 = x.data.new(self.args.num_layers, batch_size, self.args.hidden_size).fill_(0).float()
           # (D*num_layers, batch_size, hidden_size) 
            c_0 = x.data.new(self.args.num_layers, batch_size, self.args.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        # LSTM
        # output.shape=(batch_size, seq_len, D*output_size) 
        # h_0.shape   =(D*num_layers, batch_size, h_out)
        # c_0.shape   =(D*num_layers, batch_size, h_cell)
        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))
        
        # 获取 LSTM 输出的维度信息
        batch_size, seq_len, hidden_size = output.shape
        
        # 将 output 变成 (batch_size * seq_len, hidden_size)
        output = output.reshape(-1, hidden_size)

        # 全连接层
        output = self.linear(output)  # (batch_size * seq_len, 1)
        # TODO output = output.reshape(seq_len, batch_size, -1)  # 转换维度用于输出
        output = output.reshape(batch_size, seq_len, -1).permute(1, 0, 2)  # 转换维度用于输出

        # 返回最后一个时间片的数据
        # output = output[: -1, :]
        output = output[-1]

        return output


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.output_dim = 1 if args.features in ["MS", "S"] else args.feature_size
        self.input_projection = nn.Linear(args.feature_size, args.hidden_size, bias=True)
        self.activation = nn.ReLU()
        self.lstm = nn.LSTM(args.hidden_size, args.hidden_size, args.num_layers, bias=True, batch_first=True)
        self.horizon_projection = nn.Linear(args.hidden_size, args.pred_len * self.output_dim, bias=True)
 
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.activation(self.input_projection(x))
        _, (hidden_state, _) = self.lstm(x)
        final_hidden = hidden_state[-1]
        outputs = self.horizon_projection(final_hidden)

        return outputs.reshape(batch_size, self.args.pred_len, self.output_dim)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
