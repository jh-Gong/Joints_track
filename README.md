# 人体姿态的时间序列追踪

本项目是 [Learnable Triangulation of Human Pose](https://github.com/karfly/learnable-triangulation-pytorch) 项目的扩展，专注于利用神经网络在时间维度上预测和优化人体关节的3D坐标。

## 项目概述

该项目采用基于Transformer的模型，根据过去一系列的人体姿态数据，预测未来的人体姿态。模型输入为一系列的根节点位置和关节旋转（以四元数表示），输出为预测的未来根节点位置和关节旋转。

## 环境准备

### 依赖安装

首先，请确保您已安装Python 3.8或更高版本。然后，通过pip安装所需的依赖项：

```bash
pip install torch tensorboardx pandas numpy pyyaml easydict matplotlib plotly tqdm
```

### 数据集

1.  **创建数据集目录**:
    ```bash
    cd ./mvn/datasets/human36m
    mkdir source
    ```

2.  **下载数据集**:
    从[下载链接](https://pan.baidu.com/s/1oGevcGMvc3p4xWTXnRITGg?pwd=fqdt)下载数据集，并将其放入`source`目录中。最终目录结构应如下所示：
    ```
    source/
    ├── train_csv/
    │   ├── output_chunk_0.csv
    │   └── ...
    └── val_csv/
        ├── output_chunk_31.csv
        └── ...
    ```

3.  **数据集预处理**:
    此步骤包括数据筛选和数据归一化，最终生成模型所需的HDF5文件。
    *   **数据筛选**:
        ```bash
        python get_data.py
        ```
    *   **数据归一化**:
        ```bash
        python pre_process.py
        ```

### 配置文件

项目在 `experiments/human36m/` 目录中提供了默认的训练和评估配置文件 (`.yaml`)。您可以根据需要复制并修改这些文件来创建新的实验配置。

## 使用方法

### 训练

您可以使用 `main.py` 脚本来启动训练过程。

```bash
python main.py --config ./experiments/human36m/train/human36m_train_ex.yaml --logdir ./logs
```
*   `--config`: 指定训练配置文件的路径。
*   `--logdir`: 指定用于存储日志、模型权重和可视化结果的目录。

### 评估

#### 评估（使用main.py）

您也可以使用 `main.py` 脚本，在训练流程中或独立地对验证集或训练集进行评估。

```bash
python main.py --eval --eval_dataset val --config ./experiments/human36m/eval/human36m_eval_ex.yaml
```
*   `--eval`: 激活评估模式。
*   `--eval_dataset`: 指定要评估的数据集 (`train` 或 `val`)。

#### 独立评估（使用eval.py）

对于更灵活的评估，您可以使用 `eval.py` 脚本。这允许您在任意CSV数据上评估指定的模型。

```bash
python eval.py \
    --config /path/to/your/eval_config.yaml \
    --model_path /path/to/your/model_weights.pth \
    --data_path /path/to/your/evaluation_data.csv \
    --visualize
```
*   `--config`: 评估时使用的模型和选项配置文件。
*   `--model_path`: 要评估的预训练模型权重文件 (`.pth`) 的路径。
*   `--data_path`: 包含待评估姿态数据的CSV文件路径。
*   `--visualize`: (可选) 评估结束后，启动一个3D可视化窗口来播放预测的动作。

### 训练过程可视化

您可以使用TensorBoard来监控训练过程中的损失变化。

```bash
tensorboard --logdir ./logs
```

训练过程中生成的3D姿态可视化HTML文件也会保存在 `--logdir` 指定的目录中。
