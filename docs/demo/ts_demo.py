import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
 
# 随机数种子
np.random.seed(0)
 
 
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)
 
 
def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
 
"""
数据定义部分
"""
true_data = pd.read_csv('ETTh1.csv')  # 填你自己的数据地址,自动选取你最后一列数据为特征列
 
 
target = 'OT'  # 添加你想要预测的特征列
test_size = 0.15  # 训练集和测试集的尺寸划分
train_size = 0.85  # 训练集和测试集的尺寸划分
pre_len = 4  # 预测未来数据的长度
train_window = 32  # 观测窗口
 
# 这里加一些数据的预处理, 最后需要的格式是pd.series
true_data = np.array(true_data[target])
 
# 定义标准化优化器
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))
 
# 训练集和测试集划分
train_data = true_data[:int(train_size * len(true_data))]
test_data = true_data[-int(test_size * len(true_data)):]
print("训练集尺寸:", len(train_data))
print("测试集尺寸:", len(test_data))
 
# 进行标准化处理
train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))
 
# 转化为深度学习模型需要的类型Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized)
test_data_normalized = torch.FloatTensor(test_data_normalized)
 
 
def create_inout_sequences(input_data, tw, pre_len):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + 4) > len(input_data):
            break
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq
 
 
# 定义训练器的的输入
train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len)
test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)
 
# 创建数据集
train_dataset = TimeSeriesDataset(train_inout_seq)
test_dataset = TimeSeriesDataset(test_inout_seq)
 
# 创建 DataLoader
batch_size = 32  # 你可以根据需要调整批量大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
 
 
class GRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, output_dim=1, pre_len= 4):
        super(GRU, self).__init__()
        self.pre_len = pre_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # 替换 LSTM 为 GRU
        self.gru = nn.GRU(input_dim, hidden_dim,num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        h0_gru = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
 
        out, _ = self.gru(x, h0_gru)
 
        out = self.dropout(out)
 
        # 取最后 pre_len 时间步的输出
        out = out[:, -self.pre_len:, :]
 
        out = self.fc(out)
        out = self.relu(out)
        return out
 
 
lstm_model = GRU(input_dim=1, output_dim=1, num_layers=2, hidden_dim=train_window, pre_len=pre_len)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)
epochs = 20
Train = True  # 训练还是预测
 
if Train:
    losss = []
    lstm_model.train()  # 训练模式
    for i in range(epochs):
        start_time = time.time()  # 计算起始时间
        for seq, labels in train_loader:
            lstm_model.train()
            optimizer.zero_grad()
 
            y_pred = lstm_model(seq)
 
            single_loss = loss_function(y_pred, labels)
 
            single_loss.backward()
            optimizer.step()
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        losss.append(single_loss.detach().numpy())
    torch.save(lstm_model.state_dict(), 'save_model.pth')
    print(f"模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")
 
 
else:
    # 加载模型进行预测
    lstm_model.load_state_dict(torch.load('save_model.pth'))
    lstm_model.eval()  # 评估模式
    results = []
    reals = []
    losss = []
 
    for seq, labels in test_loader:
        pred = lstm_model(seq)
        mae = calculate_mae(pred.detach().numpy(), np.array(labels))  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)
        for j in range(batch_size):
            for i in range(pre_len):
                reals.append(labels[j][i][0].detach().numpy())
                results.append(pred[j][i][0].detach().numpy())
 
    reals = scaler_test.inverse_transform(np.array(reals).reshape(1, -1))[0]
    results = scaler_test.inverse_transform(np.array(results).reshape(1, -1))[0]
    print("模型预测结果：", results)
    print("预测误差MAE:", losss)
 
    plt.figure()
    plt.style.use('ggplot')
 
    # 创建折线图
    plt.plot(reals, label='real', color='blue')  # 实际值
    plt.plot(results, label='forecast', color='red', linestyle='--')  # 预测值
 
    # 增强视觉效果
    plt.grid(True)
    plt.title('real vs forecast')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('test——results.png')