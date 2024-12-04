import os
import shutil
from typing import Tuple
import numpy as np
import inspect
import joblib
from datetime import datetime
import random
from collections import defaultdict
import time
from tqdm import tqdm
from easydict import EasyDict as edict

import torch
from torch import autograd
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter

from mvn.utils import cfg
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss
from mvn.models.basicnet import LstmModel
from mvn.datasets.human36m import Human36MMultiJointsDataset
import mvn.utils.misc as misc
from mvn.utils.pose_show_3d import save_3d_png


def setup_human36m_dataloaders(config: edict, is_train: bool):
    """
    根据配置和训练标志设置Human3.6M数据集的数据加载器。

    参数:
    - config: 包含数据集和训练配置的对象。(从yaml配置文件导入)
    - is_train: 布尔值，指示是否需要设置训练数据加载器。

    返回:
    - train_dataloader: 训练数据加载器，如果is_train为False，则为None。
    - val_dataloader: 验证数据加载器。
    """
    # 初始化训练数据加载器为None
    train_dataloader = None

    # 如果is_train为True，则设置训练数据加载器
    if is_train:
        # 加载训练数据
        train_data = joblib.load(config.dataset.train.data_path)
        # 从多个动作中创建训练数据集
        train_data_x, train_data_y = create_dataset_from_multi_actions(
            train_data, num_actions=config.opt.n_actions_per_epochs, 
            num_joints=config.opt.n_joints, seq_len=config.opt.seq_len)
        print("train data loaded!")
        # 创建训练数据集对象
        train_dataset = Human36MMultiJointsDataset(train_data_x, train_data_y)

        # 创建训练数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.shuffle,
            num_workers=config.dataset.loader_num_workers
        )

    # 加载验证数据
    val_data = joblib.load(config.dataset.val.data_path)
    # 从多个动作中创建验证数据集
    val_data_x, val_data_y = create_dataset_from_multi_actions(
        val_data, num_actions=config.opt.n_actions_per_epochs, 
        num_joints=config.opt.n_joints, seq_len=config.opt.seq_len)
    print("val data loaded!")
    # 创建验证数据集对象
    val_dataset = Human36MMultiJointsDataset(val_data_x, val_data_y)

    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.dataset.loader_num_workers
    )

    # 返回训练和验证数据加载器
    return train_dataloader, val_dataloader


def setup_dataloaders(config, is_train=True):
    """
    根据配置和训练状态设置数据加载器。

    参数:
    - config: 包含数据集配置的配置对象。
    - is_train: 布尔值，表示是否为训练模式。

    返回:
    - train_dataloader: 训练数据加载器。
    - val_dataloader: 验证数据加载器。

    此函数根据配置文件中指定的数据集类型，设置相应的数据加载器。
    如果数据集类型不是`human36m`，则抛出异常。
    """
    # 根据配置文件中指定的数据集类型设置数据加载器
    if config.dataset.kind == 'human36m':
        # 配置Human3.6M数据集的数据加载器
        train_dataloader, val_dataloader = setup_human36m_dataloaders(config, is_train)
    else:
        # 如果数据集类型未知，抛出异常
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    # 返回设置好的数据加载器
    return train_dataloader, val_dataloader

def create_dataset_from_multi_actions(data: dict, num_actions: int = 10, num_joints: int = 17, seq_len: int = 5) -> Tuple[np.array, np.array]:
    """
    从包含多个动作的数据字典中创建数据集。
    
    参数:
    - data: 包含多个动作数据的字典，每个动作对应一个键值对。
    - num_actions: 用于创建数据集的动作数量，默认为10。
    - num_joints: 每个动作中的关节数量，默认为17。
    - seq_len: 每个数据样本的序列长度，默认为5。
    
    返回:
    - all_dataset_x: 所有动作数据的输入特征数组。
    - all_dataset_y: 所有动作数据的标签数组。
    """
    # 初始化用于存储所有数据集的列表
    all_dataset_x, all_dataset_y = [], []
    
    # 获取所有动作键的列表，并确保只选取前num_actions个动作
    action_key_list_all = list(data.keys())
    action_key_list_all = [key for key in action_key_list_all if key.startswith("action")]
    
    # 检查num_actions是否超出实际动作数量
    if num_actions > len(action_key_list_all):
        action_keys = action_key_list_all
        print(f"n_actions_per_epochs settings out of index({len(action_key_list_all)})")
    else:
        action_keys = action_key_list_all[: num_actions]
    
    # 遍历每个动作键，创建并合并数据集
    for key in action_keys:
        action_data = np.array(data[key])
        dataset_x, dataset_y = create_dataset_from_one_action(action_data, num_joints, seq_len)
        all_dataset_x.append(dataset_x)
        all_dataset_y.append(dataset_y)
    
    # 合并数据
    all_dataset_x = torch.from_numpy(np.concatenate(all_dataset_x, axis=0)).float()
    all_dataset_y = torch.from_numpy(np.concatenate(all_dataset_y, axis=0)).float()
    
    return all_dataset_x, all_dataset_y

def create_dataset_from_one_action(data: np.array, num_joints: int = 17, seq_len: int = 5) -> Tuple[np.array, np.array]:
    """
    参数：
    - data: 某一个姿态的连续帧关节坐标点列表。shape: [num_frames, num_joints, 3]
    - num_joints: 要使用的关节数量。
    - seq_len: 时间序列长度。

    返回一对对应的元组，分别为输入和输出。
    shape: ([num_frames - seq_len, seq_len, num_joints * 3], [num_frames - seq_len, seq_len  + 1,  num_joints * 3])
    """
    # 检查数据规范
    if len(data.shape) != 3 or data.shape[2] != 3:
        raise ValueError("数据形状必须为 [num_frames, num_joints, 3]")
    num_frames, num_data_joints, input_size = data.shape
    if num_joints < 0 or num_joints > num_data_joints:
        current_function_name = inspect.currentframe().f_code.co_name
        raise ValueError(f"main.py: {current_function_name}(): num_joints out of index({num_data_joints})!")

    data = data.reshape(num_frames, num_data_joints * input_size)

    # 获取输入输出序列
    dataset_x, dataset_y = [], []
    for i in range(len(data) - seq_len):
        dataset_x.append(data[i: i + seq_len, :num_joints * input_size])
        dataset_y.append(data[i: i + seq_len + 1, :num_joints * input_size])
    
    return np.array(dataset_x), np.array(dataset_y)

def setup_experiment(config, model_name, is_train=True):
    """
    准备实验所需的目录和日志。

    Args:
        config: 实验配置对象，包含实验的各种配置参数。
        model_name: 模型名称，用于生成实验标题。
        is_train: 布尔值，表示是否为训练模式。默认为True。

    Returns:
        experiment_dir: 实验目录路径。
        writer: TensorBoard SummaryWriter对象，用于记录训练或评估过程中的信息。
    """
    # 根据是否为训练模式，设置前缀为空或"eval_"
    prefix = "" if is_train else "eval_"

    # 根据配置中的标题和模型名称生成实验标题
    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    # 添加前缀以区分训练和评估
    experiment_title = prefix + experiment_title

    # 生成实验名称，包含标题和当前时间
    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%m-%d-%H-%M-%S.%Y"))
    print("Experiment name: {}".format(experiment_name))

    # 创建实验目录
    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 创建检查点目录，用于保存模型权重
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 将实验配置文件复制到实验目录中
    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # 创建TensorBoard SummaryWriter对象
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # 记录配置信息到TensorBoard
    writer.add_text(misc.config_to_str(config), "config", 0)

    # 返回实验目录路径和SummaryWriter对象
    return experiment_dir, writer

def one_epoch(model, criterion, opt, config, dataloader, device, epoch, is_train=True, experiment_dir=None, writer=None):
    """
    训练或验证模型一个epoch。

    参数:
    - model: 使用的模型。
    - criterion: 损失函数。
    - opt: 优化器。
    - config: 配置对象，包含模型和训练配置。
    - dataloader: 数据加载器。
    - device: 设备，'cuda' 或 'cpu'。
    - epoch: 当前epoch编号。
    - is_train: 布尔值，表示是否为训练模式。
    - experiment_dir: 实验目录路径，用于保存结果。
    - writer: TensorBoard writer，用于记录训练过程。

    返回:
    无
    """
    # 根据训练或验证状态设置名称
    name = "train" if is_train else "val"
    # 初始化总损失
    total_loss = 0.0

    # 根据训练或验证状态设置模型状态
    if is_train:
        model.train()
    else:
        model.eval()

    # 确认梯度计算是否使用
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        # 初始化进度条
        iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{name} Epoch {epoch + 1}/{config.opt.n_epochs}")

        # 遍历数据集中的每个批次
        for batch_idx, batch in iterator:
            with autograd.detect_anomaly():
                # 检查批次是否为空
                if batch is None:
                    print("Found None batch")
                    continue

                # 预处理batch
                batch_x = batch[0].to(device)
                batch_y = batch[1].to(device)

                # 前向传播和反向梯度计算
                outputs = model(batch_x) 
                keypoints_binary_validity = torch.ones_like(outputs)
                keypoints_binary_validity[:, -1, ...] = config.opt.pre_frame_weight
                loss = criterion(outputs, batch_y, keypoints_binary_validity)
                if is_train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # 累加损失
                total_loss += loss.item()

                # 使用TensorBoard记录损失
                if writer is not None:
                    writer.add_scalar(f'Loss/batch_{name}', loss.item(), epoch * len(dataloader) + batch_idx)
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}], Loss: {avg_loss:.4f}')

        # 记录每个epoch的平均损失
        if writer is not None:
            writer.add_scalar(f'Loss/epoch_{name}', avg_loss, epoch)

    # 定期保存3D图形
    if config.opt.save_3d_png and epoch % config.opt.save_3d_png_freq == 0 and config.opt.n_joints == 17 :
        print("Saving 3d png...")
        save_3d_png(model, device, dataloader, experiment_dir, name, epoch)

    
def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config
    config = cfg.load_config(args.config)

    model = {
        "lstm": LstmModel
    }[config.model.name](3 * config.opt.n_joints, config.model.n_hidden_layer, 3 * config.opt.n_joints, config.model.n_layers).to(device)

    if (config.model.init_weights):
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load(config.model.checkpoint, weights_only=True))

    # criterion
    criterion_class = {
        "mse": KeypointsMSELoss,
        "mse_smooth": KeypointsMSESmoothLoss,
        "mae": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "mse_smooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer
    opt = None
    if not args.eval:
        model_total = sum([param.nelement() for param in model.parameters()])  # 计算模型参数
        print("Number of model_total parameter: %.8fM" % (model_total / 1e6))
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader = setup_dataloaders(config, is_train=True if not args.eval else False) 

    # experiment
    experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    if not args.eval:
        # train loop
        for epoch in range(config.opt.n_epochs):
            one_epoch(model, criterion, opt, config, train_dataloader, device, epoch, is_train=True, experiment_dir=experiment_dir, writer=writer)
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

    else:
        if args.eval_dataset == 'train':
            one_epoch(model, criterion, opt, config, train_dataloader, device, 0, is_train=False, experiment_dir=experiment_dir, writer=writer)
        else:
            one_epoch(model, criterion, opt, config, val_dataloader, device, 0, is_train=False, experiment_dir=experiment_dir, writer=writer)



if __name__ == '__main__':
    work_directory = os.path.dirname(os.path.abspath(__file__))
    args = cfg.parse_args(work_directory)
    print("args: {}".format(args))
    main(args)
