# -*- coding: utf-8 -*-

# ***************************************************
# * File        : RNN.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-28
# * Version     : 0.1.052816
# * Description : description
# * Link        : https://weibaohang.blog.csdn.net/article/details/128619443
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import torch
import torch.nn as nn


class Model_v1(nn.Module):

    def __init__(self, feature_size, hidden_size, num_layers, output_size) -> None:
        super(Model_v1, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # rnn
        self.rnn = nn.RNN(
            input_size = feature_size, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            bias = True,
            batch_first = True
        )
        # fc layer
        self.linear = nn.Linear(
            in_features = hidden_size, 
            out_features = output_size,
            bias = True,
        )

    def forward(self, x, hidden = None):
        batch_size = x.shape[0]
        # 初始化隐藏层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        # RNN
        output, h_n = self.rnn(x, h_0)
        # 全连接层
        output = self.linear(output)

        output = output[:, -1, :]
        
        return output


class Model(nn.Module):

    def __init__(self, args) -> None:
        super(Model, self).__init__()

        self.args = args
        # hideen layer
        self.hidden = nn.Linear(args.feature_size, args.hidden_size, bias=True)
        # relu
        self.relu = nn.ReLU()
        # rnn: [batch_size,seq_len,hidden_size]
        self.rnn = nn.RNN(args.hidden_size, args.hidden_size, args.num_layers, bias=True, batch_first=True)
        # output size
        output_size = 1 if (args.features == "MS" or args.features == "S") else args.feature_size
        # fc layer
        self.linear = nn.Linear(args.hidden_size, output_size, bias=True)

    def forward(self, x):
        # logger.info(f"debug::x.device: {x.device}")
        
        # [batch_size, seq_len, feature_size]
        batch_size, seq_len, feature_size = x.shape
        # logger.info(f"debug::batch_size: {batch_size}, seq_len: {seq_len}, feature_size: {feature_size}")
        
        # [batch_size, seq_len, hidden_size]
        x_concat = self.hidden(x)
        # logger.info(f"debug::x_concat.shape: {x_concat.shape}")
        
        # [batch_size, seq_len-1, hidden_size]
        H = torch.zeros(batch_size, seq_len-1, self.args.hidden_size).to(x.device)
        # logger.info(f"debug::H.shape: {H.shape}")
        
        # [num_layers, batch_size, hidden_size]
        h_t = torch.zeros(self.args.num_layers, batch_size, self.args.hidden_size).to(x.device)
        # logger.info(f"debug::h_t.shape: {h_t.shape}")
        
        for t in range(seq_len):
            # [batch_size, 1, hidden_size]
            x_t = x_concat[:, t, :].view(batch_size, 1, -1)
            # ht: [num_layers, batch_size, hidden_size]
            out, h_t = self.rnn(x_t, h_t)
            # [batch_size, hidden_size]
            htt = h_t[-1, :, :]
            if t != seq_len - 1:
                H[:, t, :] = htt
        
        # [batch_size, seq_len-1, hidden_size]
        H = self.relu(H)
        # logger.info(f"debug::H.shape: {H.shape}")
        
        # [batch_size, hidden_size, output_size]
        x = self.linear(H)
        # logger.info(f"debug::x.shape: {x.shape}")

        return x[:, -self.args.pred_len:, :]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
