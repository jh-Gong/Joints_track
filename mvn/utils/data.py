from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import h5py
from tqdm import tqdm
import numpy as np
import torch
from typing import Tuple

from mvn.datasets.human36m import Human36MMultiJointsDataset

def load_hdf5_data(data_path):
    hdf5_data = {}
    scaling_info = {}

    try:
        with h5py.File(data_path, 'r') as h5file:
            # 缩放信息
            scaling_group = h5file['scaling_info']
            for key in scaling_group.keys():
                scaling_info[key] = scaling_group[key][()]
            scaling_info['mode'] = scaling_info['mode'].decode('utf-8')
            # 数据信息
            subjects = [key for key in h5file.keys() if key != 'scaling_info']
            for subject in tqdm(subjects, desc=f"Subjects"):
                if subject not in hdf5_data:
                    hdf5_data[subject] = {}
                subject_group = h5file[subject]

                for action in tqdm(subject_group.keys(), desc=f"Actions in S{subject}", leave=False):
                    if action not in hdf5_data[subject]:
                        hdf5_data[subject][action] = {}
                    action_group = subject_group[action]

                    for subaction in action_group.keys():
                        if subaction not in hdf5_data[subject][action]:
                            hdf5_data[subject][action][subaction] = {}
                        subaction_group = action_group[subaction]

                        for video_id in subaction_group.keys():
                            if video_id not in hdf5_data[subject][action][subaction]:
                                hdf5_data[subject][action][subaction][video_id] = {}
                            video_group = subaction_group[video_id]

                            joints_3d = video_group['joints_3d'][:]
                            joints_vis = video_group['joints_vis'][:]

                            hdf5_data[subject][action][subaction][video_id]['joints_3d'] = joints_3d
                            hdf5_data[subject][action][subaction][video_id]['joints_vis'] = joints_vis
        return hdf5_data, scaling_info
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return

def setup_dataloaders(config, is_train=True):
    """
    根据配置和训练状态设置数据加载器。

    参数:
    - config: 包含数据集配置的配置对象。
    - is_train: 布尔值，表示是否为训练模式。

    返回:
    - train_dataloader: 训练数据加载器。
    - val_dataloader: 验证数据加载器。
    - scaling_info: 数据集缩放信息。

    此函数根据配置文件中指定的数据集类型，设置相应的数据加载器。
    如果数据集类型不是`human36m`，则抛出异常。
    """
    # 根据配置文件中指定的数据集类型设置数据加载器
    if config.dataset.kind == 'human36m':
        # 配置Human3.6M数据集的数据加载器
        train_dataloader, val_dataloader, train_scaling_info, val_scaling_info = setup_human36m_dataloaders(config, is_train)
    else:
        # 如果数据集类型未知，抛出异常
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    # 返回设置好的数据加载器
    return train_dataloader, val_dataloader, train_scaling_info, val_scaling_info

def setup_human36m_dataloaders(config: edict, is_train: bool):
    """
    根据配置和训练标志设置Human3.6M数据集的数据加载器。

    参数:
    - config: 包含数据集和训练配置的对象。(从yaml配置文件导入)
    - is_train: 布尔值，指示是否需要设置训练数据加载器。

    返回:
    - train_dataloader: 训练数据加载器，如果is_train为False，则为None。
    - val_dataloader: 验证数据加载器。
    - val_scaling_info: 数据集的缩放信息。
    """
    # 初始化训练数据加载器为None
    train_dataloader = None
    train_scaling_info =None

    # 如果is_train为True，则设置训练数据加载器
    if is_train:
        # 加载训练数据
        train_data, train_scaling_info = load_hdf5_data(config.dataset.train.data_path)
        # 创建训练数据集对象
        train_dataset = Human36MMultiJointsDataset(train_data, seq_len=config.opt.seq_len)

        # 创建训练数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=True,
            num_workers=config.dataset.loader_num_workers
        )

    # 加载验证数据
    val_data, val_scaling_info = load_hdf5_data(config.dataset.val.data_path)
    # 创建验证数据集对象
    val_dataset = Human36MMultiJointsDataset(val_data, seq_len=config.opt.seq_len)

    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.batch_size,
        shuffle=False,
        num_workers=config.dataset.loader_num_workers
    )

    # 返回训练和验证数据加载器
    return train_dataloader, val_dataloader, train_scaling_info, val_scaling_info

