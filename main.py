import os
import shutil
from datetime import datetime
from tqdm import tqdm
import json
import argparse
from typing import Union, Dict, Optional, Tuple

import torch
from torch import autograd
from torch import optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from mvn.utils import cfg
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, QuaternionAngleLoss
from mvn.models.builder import build_model
from mvn.utils.data import setup_dataloaders
import mvn.utils.misc as misc
from mvn.utils.visual import save_3d_png, get_keypoints_error

def setup_experiment(config: edict, model_name: str, is_train: bool = True) -> Tuple[str, SummaryWriter]:
    """
    准备实验所需的目录和日志。

    Args:
        config (edict): 实验配置对象。
        model_name (str): 模型名称，用于生成实验标题。
        is_train (bool, optional): 是否为训练模式。默认为True。

    Returns:
        Tuple[str, SummaryWriter]: 实验目录路径和TensorBoard SummaryWriter对象。
    """
    prefix = "" if is_train else "eval_"
    experiment_title = (config.title + "_" + model_name) if config.title else model_name
    experiment_title = prefix + experiment_title

    # 获取 'args' 变量
    args = config.args

    experiment_name = f'{experiment_title}@{datetime.now().strftime("%m-%d-%H-%M-%S.%Y")}'
    print(f"实验名称: {experiment_name}")

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))
    writer.add_text("config", misc.config_to_str(config), 0)

    return experiment_dir, writer

def one_epoch(
    model: Module,
    criterion: Module,
    criterion_rotations: Module,
    opt: Optional[Optimizer],
    config: edict,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    is_train: bool = True,
    experiment_dir: Optional[str] = None,
    writer: Optional[SummaryWriter] = None
) -> Optional[Dict]:
    """
    训练或验证模型一个epoch。

    Args:
        model (Module): 使用的模型。
        criterion (Module): 关键点损失函数。
        criterion_rotations (Module): 旋转损失函数。
        opt (Optional[Optimizer]): 优化器。
        config (edict): 配置对象。
        dataloader (DataLoader): 数据加载器。
        device (torch.device): 'cuda' 或 'cpu'。
        epoch (int): 当前epoch编号。
        is_train (bool, optional): 是否为训练模式。默认为True。
        experiment_dir (Optional[str], optional): 实验目录路径。默认为None。
        writer (Optional[SummaryWriter], optional): TensorBoard writer。默认为None。

    Returns:
        Optional[Dict]: 如果是验证模式且需要保存关键点误差，则返回误差字典。
    """
    name = "train" if is_train else "val"
    total_loss = 0.0
    model.train() if is_train else model.eval()

    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = tqdm(dataloader, desc=f"{name} Epoch {epoch + 1}/{config.opt.n_epochs}", total=len(dataloader))
        for batch_idx, batch in enumerate(iterator):
            if batch is None:
                print("发现空批次，跳过。")
                continue

            batch_root_x = batch[0]['root'].to(device)
            batch_rotations_x = batch[0]['rotations'].to(device)
            batch_root_y = batch[1]['root'].to(device)
            batch_rotations_y = batch[1]['rotations'].to(device)

            root_out, rotations_out = model(batch_root_x, batch_rotations_x)
            root_loss = criterion(root_out, batch_root_y)
            rotations_loss = criterion_rotations(rotations_out, batch_rotations_y)
            loss = root_loss + config.opt.rotation_loss_weight * rotations_loss

            if is_train and opt:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss += loss.item()

            if writer:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar(f'Loss/root_{name}', root_loss.item(), global_step)
                writer.add_scalar(f'Loss/rotations_{name}', rotations_loss.item(), global_step)
                writer.add_scalar(f'Loss/batch_{name}', loss.item(), global_step)

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}], {name} 平均损失: {avg_loss:.10f}')

        if writer:
            writer.add_scalar(f'Loss/epoch_{name}', avg_loss, epoch)

    if config.opt.save_3d_png and epoch % config.opt.save_3d_png_freq == 0 and config.opt.n_joints == 17:
        print("保存3D可视化...")
        save_3d_png(model, device, dataloader, experiment_dir, name, epoch)

    if not is_train and config.opt.save_keypoints_error and epoch % config.opt.save_keypoints_error_freq == 0:
        return get_keypoints_error(model, device, dataloader)

    return None

def main(args: argparse.Namespace):
    """
    主函数，用于训练或评估模型。

    Args:
        args (argparse.Namespace): 命令行参数。
    """
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = cfg.load_config(args.config)
    config.args = args  # 将args附加到config中

    model = build_model(config, device)

    criterion_class = {
        "mse": KeypointsMSELoss,
        "mse_smooth": KeypointsMSESmoothLoss,
        "mae": KeypointsMAELoss
    }[config.opt.criterion]
    criterion = criterion_class(config.opt.mse_smooth_threshold) if config.opt.criterion == "mse_smooth" else criterion_class()
    criterion_rotations = QuaternionAngleLoss()

    opt = None
    if not args.eval:
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {model_total_params / 1e6:.6f}M")
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    print("加载数据...")
    is_train_mode = not args.eval or args.eval_dataset == 'train'
    train_dataloader, val_dataloader = setup_dataloaders(config, is_train=is_train_mode)

    experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    if not args.eval:
        for epoch in range(config.opt.n_epochs):
            one_epoch(model, criterion, criterion_rotations, opt, config, train_dataloader, device, epoch, is_train=True, experiment_dir=experiment_dir, writer=writer)
            loss_dict = one_epoch(model, criterion, criterion_rotations, None, config, val_dataloader, device, epoch, is_train=False, experiment_dir=experiment_dir, writer=writer)

            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", f"{epoch:04d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

            if loss_dict:
                with open(os.path.join(checkpoint_dir, "loss.json"), 'w') as f:
                    json.dump(loss_dict, f, indent=4)
    else:
        dataloader_to_eval = train_dataloader if args.eval_dataset == 'train' else val_dataloader
        loss_dict = one_epoch(model, criterion, criterion_rotations, None, config, dataloader_to_eval, device, 0, is_train=False, experiment_dir=experiment_dir, writer=writer)

        result_dir = os.path.join(experiment_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        if loss_dict:
            with open(os.path.join(result_dir, "final_loss.json"), 'w') as f:
                json.dump(loss_dict, f, indent=4)

if __name__ == '__main__':
    work_directory = os.path.dirname(os.path.abspath(__file__))
    args = cfg.parse_args(work_directory)
    print(f"参数: {args}")
    main(args)
    print("完成。")
