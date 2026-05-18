---
source_type: web
created_at: 2026-01-01  # default placeholder, actual date unknown
topics: [tsproj_dl]
status: summarized
---


<details><summary>目录</summary><p>

- [项目说明](#项目说明)
    - [步骤 1：确定代码框架](#步骤-1确定代码框架)
    - [步骤 2：定义命令行解析](#步骤-2定义命令行解析)
    - [步骤 3：确定调参工具](#步骤-3确定调参工具)
    - [步骤 4：减少随机性](#步骤-4减少随机性)
- [项目优化](#项目优化)
</p></details><p></p>

# 项目说明

## 步骤 1：确定代码框架

首先确定好具体任务，然后根据任务选择合适的框架，如 `PyTorch Lightning` 或 `MMDection`。
如果框架有默认目录，则遵守。否则可以创建适合自己的目录，一般而言目录推荐如下：

* `general`：常见的训练过程、保存加载模型过程，与具体任务相关的代码
* `layers`：模型定义、损失函数等
* `models`: 模型定义、损失函数等
* `experiments`：具体任务的训练流程、数据读取和验证过程

```
general/
│   train.py
│   task.py
│   mutils.py
layers/
experiments/
│   task1/
│        train.py
│        task.py
│        eval.py
│        dataset.py
│   task2/
│        train.py
│        task.py
│        eval.py
│        dataset.py
```

## 步骤 2：定义命令行解析

Notebook 虽然很好用，但是具体 `.py` 代码实际运行和管理更加方便。所以命令行解析就非常关键。
可以选择自己喜欢的参数解析器，在命令行中一般推荐加入**学习率**、**batch size**、**seed** 等超参数。

```bash
$ python train.py --learning ... --seed ... --hidden_size ...
```

## 步骤 3：确定调参工具

在调试和训练模型的过程中，肯定需要多次训练，此时 TensorBoard 可以非常好的管理实验日志。

调参是非常乏味的，比较重要的是确定好**学习率**和 **batch size**。
**学习率**和**优化器**有非常多的选择，SGD 是一个比较好的开始。
一般而言模型越深，学习率越小；batch size 越大，学习率越大。

## 步骤 4：减少随机性

深度学习模型有一定的随机性，模型是否可复现非常重要。在比赛期间，
非常推荐提前把不同 fold 的次序存储到文件，减少随机性。

把配置文件、模型权重、日志文件保存好，这样每次都可以进行实验对比。

* PyTorch 设置 SEED

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

* TF 1.X 设置 SEED

```python
from tfdeterminism import patch
patch()
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)
```

* TF 2.X 设置 SEED

```python
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

# 项目优化

1. 本项目主要作用是实现 “基于神经网络模型的时间序列预测”快速测试、训练、推理框架
2. 项目中包含了多类模型，具体分类参考 `models` 目录下模型的子目录分类：
    - 有些模型已经比较完整，可以直接使用
    - 有些模型没有完成，需要进行补充
    - 有些模型只是建立了空脚本，需要进行调研和编写补充
3. 项目中模型的主要构建模块在 `layers` 中，主要包含了神经网络的定义，目前没有进行分类，是否需要分类还有待考虑
4. 项目的数据处理流程在 `data_provider` 目录中，由于对各类模型输入数据格式、处理流程的不同，目前进行了简单分类
    - `RNNs_type` 为 RNNs 模型的数据处理流程
    - `TFs_type` 为 TFs 模型的数据处理流程
    - `todo` 为未完成的模型的数据处理流程，是 `RNNs_type` 的未完成部分，需要进行整合
5. 目录 `dataset` 中包含了很多公开数据集，用于模型进行测试；
6. 目录 `exp` 中包含了模型运行的主流程，包括训练、验证、推理等流程
    - `exp_basic.py` 是一个接口类型脚本，用于定义模型的主流程，目前的想法是各种模型都可以集成这个脚本
    - `exp_forecasting.py` 是一个数据集类型脚本，用于定义模型的主流程，目前是作为 RNNs 模型的主流程
    - `exp_long_term_forecasting.py` 和 `exp_short_term_forecasting.py` 是两个时间序列类型脚本，用于定义模型的主流程，目前是作为 TFs 模型的主流程
    - 其他类型模型的主流程可以在这个目录中进行定义，如果可以使用目前已有的脚本，可以直接使用，如果需要自己进行定义，可以在这个目录中进行定义
7. 目录 `docs` 中包含了一些可读的文档，用于保存帮助进行时间序列预测研究的相关文档，其他有用的文档可以保存在这个目录中
8. 目录 `scripts` 中包含了一些可执行的脚本，用于进行时间序列预测研究的快速测试，其他有用的脚本可以保存在这个目录中，目前以数据集进行分类构建
9. `logs` 和 `saved_results`（希望重命名为 `results`）目录用于保存模型的训练结果，以及模型的推理结果
10. 目前的 `utils` 目录是一个单独的 git 仓库，包含了其他算法项目的依赖，作为一个统一的工具库供多个项目使用。优化效果：
   新建 `utils` 仓库，作为本项目单独的依赖库，从 git 仓库 `utils` 中提取出目前项目的依赖。最后将 git 仓库 `utils` 删除。
