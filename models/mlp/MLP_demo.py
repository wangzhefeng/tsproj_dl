import pandas as pd 
import requests
import torch
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt


#- 可直接从网站读取数据
# #下载sunspots数据
# url = 'http://sidc.be/silso/DATA/SN_m_tot_V2.0.csv' 
# r = requests.get(url)
# data = pd.read_csv(url, sep=';', header=None) 
#- 也可从 http://sidc.be/silso/DATA/SN_m_tot_V2.0.csv 下载到本地硬盘上读取
path = './SN_m_tot_V2.0.csv'
data = pd.read_csv(path, sep=';', header=None) # 读取数据到dataFrame格式变量中
sunspots = data[3]  # 数据第3列是太阳黑子的月平均数据 


#- 搭建网络框架
class MLP(nn.Module):   # 定义多层感知器类，该类继承了nn.Module类
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):  # 当创建实例时，__init__() 方法被自动调用为创建的实例增加实例属性，self代表类的实例
    super().__init__() # super().__init__() 就是调用父类 (nn.Module) 的__init__()方法
    self.layers = nn.Sequential(      # Sequential内的各层顺序连接
      nn.Linear(in_features=3, out_features=4),   # 注意上一层的out_features必须与下层的in_features一致
      nn.ReLU(),          
      nn.Linear(in_features=4, out_features=3), 
      nn.ReLU(),
      nn.Linear(in_features=3, out_features=1)
    )

  def forward(self, x): # forward方法一般用来定义各个层之间的连接关系。本例中，各层的连接关系已经由__init__()方法中的Sequential定义，因此forward什么也不做，只是向神经网络提供数据，并返回输出。
    '''Forward pass'''
    return self.layers(x)


#- Set fixed random number seed
torch.manual_seed(42)  


#- Initialize the MLP
mlp = MLP()
mlp = mlp.double() # 以免程序报错：RuntimeError: expected scalar type Double but found Float


#- Define the loss function and optimizer
loss_function = nn.MSELoss()   # 均方误差 
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)


#- 将数据划分为训练集和测试集
xtrain = sunspots[:3281]  
xtest = sunspots[3281:] 


#- 将训练数据按网络结构分为输入数据和输出数据。本例中输入为t， t-1， t-2时刻的值，输出为t+1时刻的值。
x = pd.concat([xtrain.shift(2), xtrain.shift(1), xtrain], axis=1)
x.dropna(how='any', inplace=True) # 去掉任何包含NaN的行
x.drop(x.index[-1], inplace=True) # 最后一组没有对应的输出值，去掉
x = torch.from_numpy(x.values)    # 将dataFrame变量转变为tensor变量
y = xtrain.shift(-3)
y.dropna(how='any', inplace=True)
y = torch.from_numpy(y.values)


#- 将数据分批 
batch_size = 100
torch_dataset = Data.TensorDataset(x, y) # 将x,y一一对应起来，形成数据对(x,y)
trainloader = Data.DataLoader(   # 将数据分批，shuffle=True 打乱（x,y）数据对之间原有的时间顺序
    dataset=torch_dataset, batch_size=batch_size, shuffle=False
    )


#- Run the training loop
epoch_size = 170
epoch_loss = []  # 存放每一轮的平均损失值
for epoch in range(0, epoch_size): 

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):
        
        # Get inputs
        inputs, targets = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = mlp(inputs)
        
        # Compute loss
        targets=torch.reshape(targets,outputs.shape) # 将targets和outputs的shape统一，以免程序出现UserWarning: Using a target size (torch.Size([100])) that is different to the input size (torch.Size([100, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
        loss = loss_function(outputs, targets)
        
        # Perform backward pass
        loss.backward() # backward()用于计算函数的导数，在深度学习中常用于梯度下降法求损失函数关于模型参数的梯度
        
        # Perform optimization
        optimizer.step()
        
        # Loss statistics
        L = loss.item()
        current_loss += L
    
    current_loss = current_loss / (i+1)
    epoch_loss.append(current_loss)
    current_loss = 0.0


#- 查看训练结果
# Loss 随训练轮次的变化
plt.figure()
plt.plot(range(1,epoch_size+1), epoch_loss)

# in-sample prediction
plt.figure()
inpred = mlp(x)
plt.plot(y,'r-',label='obs')  
plt.plot(inpred.detach().numpy(),'b-',label='in-sample pred')
plt.legend() 


#- 向前1步预报

steps = xtest.size  # 预报步长

# 不带最新观测值的滚动预报 
obs = torch.from_numpy(xtrain[-3:].values)  # 预报起始输入值
pred1 = []
for i in range(0, steps):
    temp = mlp(obs)
    obs = torch.cat((obs[-2:],temp))   # 用预测值作为下一步预报的输入值
    pred1.append(temp.item())
plt.figure()
plt.plot(range(0,steps), xtest,'r-',label='obs')  
plt.plot(range(0,steps), pred1,'b-',label='pred1') 

# 带最新观测值的滚动预报
obs = torch.from_numpy(xtrain[-3:].values)  # 预报起始输入值
pred2 = []
for i in range(0, steps):
    temp = mlp(obs)
    obs = torch.cat((obs[-2:],torch.tensor([xtest.iloc[i]])))  # 用新的观测值作为下一步预报的预报值
    pred2.append(temp.item())
plt.plot(range(0,steps), pred2,'g-',label='pred2')
plt.legend() 