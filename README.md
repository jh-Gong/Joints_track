# 时间尺度上的人体关节追踪

本项目是对[Learnable Triangulation of Human Pose](https://github.com/karfly/learnable-triangulation-pytorch)项目的延申，主要方向是在时间层面上利用神经网络预测、优化关节的三维坐标。

## 数据集
进入mvn/datasets/human36m目录，下载所需的数据集放入source目录内，进入tools目录调用get_data.py生成本项目可用数据集，调用pre_process.py并选择模式预处理数据集。

## 使用

```
python main.py 
```
具体参数设置可在main.py内具体查看，直接使用会使用默认路径和参数。