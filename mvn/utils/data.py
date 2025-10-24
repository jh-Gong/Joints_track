r'''
Author: gjhhh 1377019164@qq.com
Date: 2024-12-07 01:03:33
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-12 22:41:58
FilePath: \Joint_track\mvn\utils\data.py
Description: 模型训练与评估的数据导入模块
'''

from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import h5py
from tqdm import tqdm

from mvn.datasets.dataset import Human36MMultiJointsDataset

def load_hdf5_data(data_path):
    """
    加载HDF5数据。

    Args:
        data_path (str): HDF5文件的路径。

    Returns:
        dict: 包含加载的数据的字典。
    """
    hdf5_data = {}

    try:
        with h5py.File(data_path, 'r') as h5file:
            # 数据信息
            subjects = list(h5file.keys())
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

                            hdf5_data[subject][action][subaction][video_id]['root'] = video_group['root'][:]
                            hdf5_data[subject][action][subaction][video_id]['rotations'] = video_group['rotations'][:]
                            hdf5_data[subject][action][subaction][video_id]['bone_lengths'] = video_group['bone_lengths'][:]

        return hdf5_data
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return

def setup_dataloaders(config, is_train=True):
    """
    根据配置和训练状态设置数据加载器。

    Args:
        config (easydict.EasyDict): 包含数据集配置的配置对象。
        is_train (bool, optional): 布尔值，表示是否为训练模式。默认为True。

    Returns:
        tuple:
            - torch.utils.data.DataLoader: 训练数据加载器。
            - torch.utils.data.DataLoader: 验证数据加载器。

    Raises:
        NotImplementedError: 如果数据集类型未知。
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

def setup_human36m_dataloaders(config: edict, is_train: bool):
    """
    根据配置和训练标志设置Human3.6M数据集的数据加载器。

    Args:
        config (easydict.EasyDict): 包含数据集和训练配置的对象。(从yaml配置文件导入)
        is_train (bool): 布尔值，指示是否需要设置训练数据加载器。

    Returns:
        tuple:
            - torch.utils.data.DataLoader: 训练数据加载器，如果`is_train`为False，则为None。
            - torch.utils.data.DataLoader: 验证数据加载器。
    """
    # 初始化训练数据加载器为None
    train_dataloader = None

    # 如果is_train为True，则设置训练数据加载器
    if is_train:
        # 加载训练数据
        train_data = load_hdf5_data(config.dataset.train.data_path)
        # 创建训练数据集对象
        train_dataset = Human36MMultiJointsDataset(train_data, seq_len=config.opt.seq_len, preload=False)

        # 创建训练数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=True,
            num_workers=config.dataset.loader_num_workers,
            pin_memory=True
        )

    # 加载验证数据
    val_data = load_hdf5_data(config.dataset.val.data_path)
    # 创建验证数据集对象
    val_dataset = Human36MMultiJointsDataset(val_data, seq_len=config.opt.seq_len, preload=False)

    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.batch_size,
        shuffle=False,
        num_workers=config.dataset.loader_num_workers,
        pin_memory=True
    )

    # 返回训练和验证数据加载器
    return train_dataloader, val_dataloader
