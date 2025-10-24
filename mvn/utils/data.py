r'''
Author: gjhhh 1377019164@qq.com
Date: 2024-12-07 01:03:33
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-12 22:41:58
FilePath: \Joint_track\mvn\utils\data.py
Description: 模型训练与评估的数据导入模块
'''
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import h5py
from tqdm import tqdm

from mvn.datasets.dataset import Human36MMultiJointsDataset

def load_hdf5_data(data_path: str) -> Optional[Dict]:
    """
    加载HDF5数据。

    Args:
        data_path (str): HDF5文件的路径。

    Returns:
        Optional[Dict]: 包含加载的数据的字典，如果加载失败则返回None。
    """
    hdf5_data = {}
    try:
        with h5py.File(data_path, 'r') as h5file:
            subjects = list(h5file.keys())
            for subject in tqdm(subjects, desc=f"正在加载 subjects"):
                if subject not in hdf5_data:
                    hdf5_data[subject] = {}
                subject_group = h5file[subject]

                for action in tqdm(subject_group.keys(), desc=f"加载 {subject} 中的动作", leave=False):
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

                            hdf5_data[subject][action][subaction][video_id]['root'] = video_group['root'][:]
                            hdf5_data[subject][action][subaction][video_id]['rotations'] = video_group['rotations'][:]
                            hdf5_data[subject][action][subaction][video_id]['bone_lengths'] = video_group['bone_lengths'][:]
        return hdf5_data
    except Exception as e:
        print(f"加载 '{data_path}' 出错: {e}")
        return None

def setup_dataloaders(config: edict, is_train: bool = True) -> Tuple[Optional[DataLoader], DataLoader]:
    """
    根据配置和训练状态设置数据加载器。

    Args:
        config (edict): 包含数据集配置的对象。
        is_train (bool, optional): 是否为训练模式。默认为True。

    Returns:
        Tuple[Optional[DataLoader], DataLoader]: 训练和验证数据加载器。

    Raises:
        NotImplementedError: 如果数据集类型未知。
    """
    if config.dataset.kind == 'human36m':
        return setup_human36m_dataloaders(config, is_train)
    else:
        raise NotImplementedError(f"未知的数据集类型: {config.dataset.kind}")

def setup_human36m_dataloaders(config: edict, is_train: bool) -> Tuple[Optional[DataLoader], DataLoader]:
    """
    设置Human3.6M数据集的数据加载器。

    Args:
        config (edict): 包含数据集和训练配置的对象。
        is_train (bool): 是否需要设置训练数据加载器。

    Returns:
        Tuple[Optional[DataLoader], DataLoader]: 训练数据加载器 (如果 is_train) 和验证数据加载器。
    """
    train_dataloader = None
    if is_train:
        train_data = load_hdf5_data(config.dataset.train.data_path)
        if train_data:
            train_dataset = Human36MMultiJointsDataset(train_data, seq_len=config.opt.seq_len)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.opt.batch_size,
                shuffle=True,
                num_workers=config.dataset.loader_num_workers,
                pin_memory=True
            )

    val_data = load_hdf5_data(config.dataset.val.data_path)
    if val_data:
        val_dataset = Human36MMultiJointsDataset(val_data, seq_len=config.opt.seq_len)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.opt.batch_size,
            shuffle=False,
            num_workers=config.dataset.loader_num_workers,
            pin_memory=True
        )
    else:
        raise ValueError("验证数据加载失败，请检查路径。")

    return train_dataloader, val_dataloader
