import os
import argparse
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

import torch
from torch import autograd
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter

from mvn.utils import cfg
from mvn.models.basicnet import LstmModel
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss
from mvn.datasets.human36m import Human36MMultiJointsDataset
import mvn.utils.misc as misc
from mvn.utils.pose_show_3d import save_3d_png


def parse_args(work_directory: str):
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")
    parser.add_argument("--logdir", type=str, default=os.path.join(work_directory, "logs"), help="Path, where logs will be stored")

    args = parser.parse_args()

    if args.config is None:
        if args.eval:
            args.config = os.path.join(work_directory, "experiments/human36m/eval/human36m_eval_ex.yaml")
        else:
            args.config = os.path.join(work_directory, "experiments/human36m/train/human36m_train_ex.yaml")

    return args

def setup_human36m_dataloaders(config, is_train):
    train_dataloader = None
    if is_train:
        # train
        train_data = joblib.load(config.dataset.train.data_path)
        train_data_x, train_data_y = create_dataset_from_multi_actions(
            train_data, num_actions=config.opt.n_actions_per_epochs, 
            num_joints=config.opt.n_joints, num_for_train=config.opt.seq_len)
        print("train data loaded!")
        train_dataset = Human36MMultiJointsDataset(train_data_x, train_data_y)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.shuffle,
            num_workers=config.dataset.loader_num_workers
        )
    
    # val
    val_data = joblib.load(config.dataset.val.data_path)
    val_data_x, val_data_y = create_dataset_from_multi_actions(
        val_data, num_actions=config.opt.n_actions_per_epochs, 
        num_joints=config.opt.n_joints, num_for_train=config.opt.seq_len)
    print("val data loaded!")
    val_dataset = Human36MMultiJointsDataset(val_data_x, val_data_y)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.dataset.loader_num_workers
    )

    return train_dataloader, val_dataloader


def setup_dataloaders(config, is_train=True):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader = setup_human36m_dataloaders(config, is_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader

def create_dataset_from_multi_actions(data: dict, num_actions: int = 10, num_joints: int = 17, num_for_train: int = 5) -> Tuple[np.array, np.array]:
    all_dataset_x, all_dataset_y = [], []
    # action_keys = random.sample(list(data.keys()), num_actions)
    action_key_list_all = list(data.keys())
    action_key_list_all = [key for key in action_key_list_all if key.startswith("action")]
    if num_actions > len(action_key_list_all):
        action_keys = action_key_list_all
        print(f"n_actions_per_epochs settings out of index({len(action_key_list_all)})")
    else:
        action_keys = action_key_list_all[: num_actions]
    for key in action_keys:
        action_data = np.array(data[key])
        dataset_x, dataset_y = create_dataset_from_one_action(action_data, num_joints, num_for_train)
        all_dataset_x.append(dataset_x)
        all_dataset_y.append(dataset_y)
    # 合并数据
    all_dataset_x = torch.from_numpy(np.concatenate(all_dataset_x, axis=0)).float()
    all_dataset_y = torch.from_numpy(np.concatenate(all_dataset_y, axis=0)).float()
    
    return all_dataset_x, all_dataset_y

def create_dataset_from_one_action(data: np.array, num_joints: int = 17, num_for_train: int = 5) -> Tuple[np.array, np.array]:
    """
    参数：
    - data: 某一个姿态的连续帧关节坐标点列表。shape: [num_frames, num_joints, 3]
    - num_joints: 要使用的关节数量。
    - num_for_train: 用于训练的帧数。

    返回一对对应的元组，分别为输入和输出。
    shape: ([num_frames - num_for_train, num_for_train, num_joints * 3], [num_frames - num_for_train, num_for_train  + 1,  num_joints * 3])
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
    for i in range(len(data) - num_for_train):
        dataset_x.append(data[i: i + num_for_train, :num_joints * input_size])
        dataset_y.append(data[i: i + num_for_train + 1, :num_joints * input_size])
    
    return np.array(dataset_x), np.array(dataset_y)

def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%Y-%m-%d.%H-%M-%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer

def one_epoch(model, criterion, opt, config, dataloader, device, epoch, is_train=True, experiment_dir=None, writer=None):
    name = "train" if is_train else "val"
    model_type = config.model.name
    total_loss = 0.0

    if is_train:
        model.train()
    else:
        model.eval()

    # 确认梯度计算是否使用
    
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{name} Epoch {epoch + 1}/{config.opt.n_epochs}")

        for batch_idx, batch in iterator:
            with autograd.detect_anomaly():

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

                total_loss += loss.item()

                # 可视化
                if writer is not None:
                    writer.add_scalar(f'Loss/batch_{name}', loss.item(), epoch * len(dataloader) + batch_idx)
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}], Loss: {avg_loss:.4f}')

        if writer is not None:
            writer.add_scalar(f'Loss/epoch_{name}', avg_loss, epoch)

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
    args = parse_args(work_directory)
    print("args: {}".format(args))
    main(args)
