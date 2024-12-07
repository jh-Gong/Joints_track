# 时间尺度上的人体关节追踪

本项目是对[Learnable Triangulation of Human Pose](https://github.com/karfly/learnable-triangulation-pytorch)项目的延伸，主要方向是在时间层面上利用神经网络预测、优化关节的三维坐标。

## 准备
### 数据集
1. 创建数据集目录
```bash
cd ./mvn/datasets/human36m
mkdir source
```
2. 下载数据集
从[下载链接](https://pan.baidu.com/s/1GVhjnypsxgOGXiK8uMSp9A?pwd=8g22)得到的数据放入source目录中
3. 数据集预处理
此步骤可以分为多个模块，包含数据筛选、数据归一化等
* 数据筛选:
```bash
python tools/get_data.py
```
* 数据归一化
首先进入tools/pre_process.py文件内切换mode为"linear"或"norm"，分别对应为线性归一化和正态分布归一化，接着运行如下指令
```bash
python tools/pre_process.py
```

### 配置文件
本项目在experiments/human36m/中内置了两个默认配置，可以在此基础上创建新配置文件。运行时会默认使用这两个配置。

## 使用
### 训练

```bash
python main.py \
--config ./experiments/human36m/train/human36m_train_ex.yaml \
--logdir ./logs
```

### 评估

```bash
python main.py \
--eval --eval_dataset val \
--config ./experiments/human36m/eval/human36m_eval_ex.yaml \
--logdir ./logs
```

### 可视化
```bash
tensorboard --logdir ./logs
```

具体参数设置可在main.py内具体查看，直接使用会使用默认路径和参数。